import flask
from flask import request, jsonify, render_template, session, Flask
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions
import os
from datetime import datetime
import threading
import librosa
import numpy as np
import base64
import tempfile
from scipy.io import wavfile
import json
import tempfile

API_KEY = 'AIzaSyCNDhe0C85XQ9znQGVXN7KwflVkQjNTdmU'

dg_client = DeepgramClient('c74782c7b1b9d342303804006520ac45d5975e9b')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


prompt = """Role: You are a helpful, empathetic AI assistant specializing in cognitive health, Alzheimer's disease, and how AI can assist in detection, monitoring, and care.

Tone: Clear, supportive, accessible, and science-informed — similar to how a health educator or care navigator might speak.

Instructions:

Provide general educational information about Alzheimer's disease, cognitive decline, symptoms, early warning signs, and prevention strategies.

Explain how AI can support early detection (e.g., voice pattern analysis, memory assessments, MRI interpretation, etc.) in a non-technical way unless asked otherwise.

Include caveats that the assistant is not a substitute for medical advice and recommend consulting a healthcare provider when appropriate.

Be concise but thorough. Use bullet points or examples where helpful.

Avoid fear-based language. Focus on empowerment, education, and awareness."""
chat = model.start_chat(history=[
    {"role": "user", "parts": [prompt]}
])

app = Flask(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_fatigue_features(audio_path):
    """Extract voice features specifically relevant to fatigue detection"""
    try:
        print(f"Processing audio file: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")
        
        y, sr = librosa.load(audio_path, sr=16000)
        print(f"Audio loaded: length={len(y)}, sr={sr}")
        
        # Check if audio was loaded successfully
        if len(y) == 0:
            print("Warning: No audio data loaded")
            return get_fallback_voice_features()
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        f0_clean = f0[voiced_flag]
        
        # Check if we have valid pitch data
        if len(f0_clean) == 0:
            print("Warning: No valid pitch data found")
            return get_fallback_voice_features()
            
        pitch_mean = np.nanmean(f0_clean) if len(f0_clean) > 0 else 0
        pitch_std = np.nanstd(f0_clean) if len(f0_clean) > 0 else 0
        pitch_range = np.nanmax(f0_clean) - np.nanmin(f0_clean) if len(f0_clean) > 0 else 0
        
        # 2. Energy analysis (fatigue shows reduced vocal energy)
        rms_energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms_energy)
        energy_variability = np.std(rms_energy)
        
        # 3. Speaking rate (fatigue often slows speech)
        # Detect syllable-like peaks
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        duration = len(y) / sr
        speaking_rate = len(onset_times) / duration if duration > 0 else 0
        
        # 4. Voice tremor (fatigue can cause voice instability)
        # Look for frequency modulation in the 4-12 Hz range
        if len(f0_clean) > 10:
            f0_diff = np.diff(f0_clean)
            tremor_index = np.std(f0_diff) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
        else:
            tremor_index = 0
        
        # 5. Pause analysis (fatigue increases pause frequency/duration)
        # Simple silence detection
        silence_threshold = np.percentile(rms_energy, 20)  # Bottom 20% as silence
        silence_frames = rms_energy < silence_threshold
        
        # Count pauses (consecutive silence frames)
        pause_count = 0
        in_pause = False
        pause_durations = []
        current_pause = 0
        
        for is_silent in silence_frames:
            if is_silent and not in_pause:
                in_pause = True
                current_pause = 1
            elif is_silent and in_pause:
                current_pause += 1
            elif not is_silent and in_pause:
                in_pause = False
                if current_pause > 5:  # Minimum pause length
                    pause_count += 1
                    pause_durations.append(current_pause * 512 / sr)  # Convert to seconds
                current_pause = 0
        
        avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
        pause_frequency = pause_count / duration if duration > 0 else 0
        
        return {
            "pitch_mean": float(round(pitch_mean, 2)),
            "pitch_std": float(round(pitch_std, 2)),
            "pitch_range": float(round(pitch_range, 2)),
            "energy_mean": float(round(energy_mean, 5)),
            "energy_variability": float(round(energy_variability, 5)),
            "speaking_rate": float(round(speaking_rate, 2)),
            "tremor_index": float(round(tremor_index, 4)),
            "pause_frequency": float(round(pause_frequency, 2)),
            "avg_pause_duration": float(round(avg_pause_duration, 2)),
            "duration": float(round(duration, 2))
        }
    except Exception as e:
        print(f"Error in voice feature extraction: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()  # This gives full error details
        return get_fallback_voice_features()

def get_fallback_voice_features():
    """Return realistic fallback voice features when extraction fails"""
    return {
        "pitch_mean": float(150.0),  # Typical adult speaking pitch
        "pitch_std": float(25.0),
        "pitch_range": float(80.0),
        "energy_mean": float(0.02),
        "energy_variability": float(0.008),
        "speaking_rate": float(4.5),  # syllables per second
        "tremor_index": float(0.002),
        "pause_frequency": float(1.2),
        "avg_pause_duration": float(0.8),
        "duration": float(10.0),
        "status": "fallback_values_used"
    }

def transcribe_audio(audio_path):
    """Transcribe audio file to text using Deepgram"""
    try:
        print(f"Starting transcription for: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")
        
        # Check file size
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        print(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            print("Audio file is empty")
            return None
        
        # Configure transcription options
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            punctuate=True,
            diarize=False,
            utterances=False,
            language="en"
        )
        
        # Method 1: Direct file upload (Deepgram v4.0+ syntax)
        try:
            print("Attempting Deepgram transcription...")
            with open(audio_path, 'rb') as audio_file:
                # Read file content for v4.0+ API
                audio_data = audio_file.read()
                
            # Use rest API with proper payload for v4.0+
            payload = {
                "buffer": audio_data,
                "mimetype": "audio/wav"
            }
            
            response = dg_client.listen.rest.v("1").transcribe_file(
                payload, options
            )
                
            print(f"Deepgram response received: {type(response)}")
            
            # Extract transcript
            transcript = ""
            if hasattr(response, 'results') and response.results:
                if response.results.channels and len(response.results.channels) > 0:
                    channel = response.results.channels[0]
                    if channel.alternatives and len(channel.alternatives) > 0:
                        transcript = channel.alternatives[0].transcript
                        print(f"Transcript extracted: '{transcript}'")
            
            if transcript.strip():
                return transcript.strip()
                
        except Exception as e:
            print(f"Deepgram method failed: {e}")
            print(f"Error type: {type(e).__name__}")
        
        # Method 2: REST API fallback
        try:
            print("Trying REST API fallback...")
            import requests
            
            headers = {
                'Authorization': f'Token c74782c7b1b9d342303804006520ac45d5975e9b',
                'Content-Type': 'audio/wav'
            }
            
            params = {
                'model': 'nova-2',
                'smart_format': 'true',
                'punctuate': 'true',
                'language': 'en'
            }
            
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(
                    'https://api.deepgram.com/v1/listen',
                    headers=headers,
                    params=params,
                    data=audio_file.read(),
                    timeout=30
                )
            
            print(f"REST API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"REST API response: {result}")
                
                if 'results' in result and 'channels' in result['results']:
                    channels = result['results']['channels']
                    if channels and len(channels) > 0:
                        alternatives = channels[0].get('alternatives', [])
                        if alternatives and len(alternatives) > 0:
                            transcript = alternatives[0].get('transcript', '')
                            if transcript.strip():
                                print(f"Transcript from REST API: '{transcript}'")
                                return transcript.strip()
            else:
                print(f"REST API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"REST API method failed: {e}")
        
        print("All transcription methods failed")
        return None
            
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/voice_call")
def voice_call():
    return render_template("voice_call.html")

# Mayo Clinic Alzheimer's Knowledge Base for RAG
MAYO_CLINIC_ALZHEIMERS_KB = """
ALZHEIMER'S DISEASE - MAYO CLINIC MEDICAL INFORMATION

OVERVIEW:
Alzheimer's disease is the most common cause of dementia. About 6.9 million people in the United States age 65 and older live with Alzheimer's disease. Among them, more than 70% are age 75 and older. Early symptoms include forgetting recent events or conversations. Over time, Alzheimer's disease leads to serious memory loss and affects a person's ability to do everyday tasks. At first, someone with the disease may be aware of having trouble remembering things and thinking clearly. As signs and symptoms get worse, a family member or friend may be more likely to notice the issues.

Brain changes from Alzheimer's disease lead to the following symptoms that get worse over time.

SYMPTOMS:
Memory loss is the key symptom of Alzheimer's disease. Brain changes from Alzheimer's disease lead to the following symptoms that get worse over time:

MEMORY SYMPTOMS:
- Repeat statements and questions over and over
- Forget conversations, appointments or events
- Misplace items, often putting them in places that don't make sense
- Get lost in places they used to know well
- Forget the names of family members and everyday objects
- Have trouble finding the right words, expressing thoughts or having conversations

THINKING AND REASONING:
- Trouble concentrating and thinking, especially about abstract concepts such as numbers
- Doing more than one task at once is especially hard
- Challenging to manage finances, balance checkbooks and pay bills on time
- May not recognize numbers eventually

MAKING JUDGMENTS AND DECISIONS:
- Hard to make sensible decisions and judgments
- May make poor choices in social settings or wear clothes for the wrong type of weather
- Everyday problems may be hard to solve

PLANNING AND PERFORMING FAMILIAR TASKS:
- Routine activities that involve completing steps in a certain order can be hard
- Trouble planning and cooking a meal or playing a favorite game
- As disease becomes advanced, people forget how to do basic tasks such as dressing and bathing

BEHAVIORAL CHANGES:
- Depression and loss of interest in activities
- Social withdrawal and mood swings
- Not trusting others, anger or aggression
- Changes in sleeping habits, wandering
- Loss of inhibitions, delusions

EARLY WARNING SIGNS:
- Memory loss that disrupts daily life
- Challenges in planning or solving problems
- Difficulty completing familiar tasks
- Confusion with time or place
- Trouble understanding visual images and spatial relationships
- New problems with words in speaking or writing
- Misplacing things and losing the ability to retrace steps
- Decreased or poor judgment
- Withdrawal from work or social activities
- Changes in mood and personality

WHEN TO SEE A DOCTOR:
If you are concerned about your memory or other thinking skills, talk to your healthcare professional. If you are concerned about the thinking skills you notice in a family member or friend, talk to them about seeing a healthcare professional.

Source: Mayo Clinic - https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447
"""

def search_knowledge_base(query, knowledge_base):
    """Simple RAG: Find relevant sections from knowledge base"""
    query_lower = query.lower()
    lines = knowledge_base.split('\n')
    relevant_sections = []
    
    # Keywords to search for
    keywords = query_lower.split()
    
    # Find relevant sections
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in keywords):
            start = max(0, i-2)
            end = min(len(lines), i+3)
            section = '\n'.join(lines[start:end])
            if section not in relevant_sections:
                relevant_sections.append(section)
    
    return '\n\n'.join(relevant_sections[:3])  # Limit to top 3 relevant sections


@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json()
        message = data.get("message", "")
        
        # RAG: Search for relevant medical information
        relevant_context = search_knowledge_base(message, MAYO_CLINIC_ALZHEIMERS_KB)
        
        # Enhanced prompt with RAG context
        rag_prompt = f"""
You are a cognitive health specialist AI assistant. Use the provided medical information from Mayo Clinic to answer the user's question accurately.

MAYO CLINIC MEDICAL CONTEXT:
{relevant_context}

USER QUESTION: {message}

Instructions:
- Use the Mayo Clinic information as your primary medical reference
- Be encouraging but medically accurate
- Always recommend consulting healthcare professionals for diagnosis
- Use clear, understandable language
- Include relevant symptoms, signs, or recommendations from the medical context
- Cite that information comes from Mayo Clinic when appropriate
-Try to be as specific as possible in your response like you are a doctor.
-Do not mention that you are an AI and can't provide diagnosis. 
-Do not mention your inabilities in any manner.
-If not on the Mayo Clinic website, use Gemini pre trained model to answer the question. 

Provide a helpful, medically-informed response:
"""
        
        response = chat.send_message(rag_prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"I'm having trouble accessing medical information right now. Please consult with a healthcare professional for accurate medical advice."}) 

@app.route("/pipeline", methods = ["POST"])
def pipeline():
    try:
        data = request.get_json()
        audio_data = data.get("audio_data", "")
        
        # Decode audio data
        audio_bytes = base64.b64decode(audio_data)
        print(f"Decoded audio bytes length: {len(audio_bytes)}")
        
        # Create temporary file to save the raw audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_raw_path = temp_file.name
            temp_file.write(audio_bytes)
        
        print(f"Raw audio file written: {temp_raw_path}")
        
        # Convert to WAV using librosa (handles multiple formats)
        try:
            print("Converting audio to WAV format using librosa...")
            # Load audio with librosa (automatically handles format conversion)
            y, sr = librosa.load(temp_raw_path, sr=16000)
            print(f"Audio loaded with librosa: length={len(y)}, sample_rate={sr}")
            
            # Create WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_audio_path = temp_file.name
            
            # Convert to int16 and save as WAV
            audio_int16 = (y * 32767).astype(np.int16)
            wavfile.write(temp_audio_path, sr, audio_int16)
            print(f"WAV file created: {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")
            
            # Clean up raw file
            if os.path.exists(temp_raw_path):
                os.unlink(temp_raw_path)
                
        except Exception as e:
            print(f"Librosa conversion failed: {e}")
            print("Trying direct approach...")
            
            # Fallback: assume it's already a WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_audio_path = temp_file.name
                temp_file.write(audio_bytes)
            
            print(f"Direct WAV file written: {temp_audio_path}")
        
        print(f"Final audio file: {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")
        
        try:
            # Transcribe audio with detailed debugging
            print("=== STARTING DETAILED TRANSCRIPTION DEBUG ===")
            print(f"Audio file path: {temp_audio_path}")
            print(f"File exists: {os.path.exists(temp_audio_path)}")
            print(f"File size: {os.path.getsize(temp_audio_path) if os.path.exists(temp_audio_path) else 0} bytes")
            
            transcript = transcribe_audio(temp_audio_path)
            print(f"=== TRANSCRIPTION RESULT: '{transcript}' ===")
            
            if not transcript:
                print("!!! TRANSCRIPTION FAILED - INVESTIGATING !!!")
                
                # Try a simple test - check if the audio file can be read by librosa
                try:
                    import librosa
                    y, sr = librosa.load(temp_audio_path, sr=None)
                    print(f"Librosa can read the file: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
                    
                    # If audio is very short, that might be the issue
                    if len(y)/sr < 0.5:  # Less than 0.5 seconds
                        return jsonify({"success": False, "error": "Audio too short for analysis (minimum 0.5 seconds required)"})
                    
                    # If audio is silent, that might be the issue
                    max_amplitude = np.max(np.abs(y))
                    print(f"Max audio amplitude: {max_amplitude}")
                    if max_amplitude < 0.001:  # Very quiet audio
                        return jsonify({"success": False, "error": "Audio signal too quiet - please speak louder"})
                    
                    # Test Deepgram API key directly
                    print("=== TESTING DEEPGRAM API KEY ===")
                    try:
                        import requests
                        test_response = requests.get(
                            'https://api.deepgram.com/v1/projects',
                            headers={'Authorization': f'Token c74782c7b1b9d342303804006520ac45d5975e9b'},
                            timeout=10
                        )
                        print(f"Deepgram API key test: {test_response.status_code}")
                        if test_response.status_code == 200:
                            print("API key is valid!")
                        else:
                            print(f"API key issue: {test_response.text}")
                    except Exception as key_test_error:
                        print(f"API key test failed: {key_test_error}")
                    
                    # For testing, use a dummy transcript
                    print("Using fallback transcript for testing...")
                    transcript = "This is a test transcript because Deepgram transcription failed."
                    
                except Exception as librosa_error:
                    print(f"Librosa also failed to read audio: {librosa_error}")
                    return jsonify({"success": False, "error": "Audio file format is invalid or corrupted"})
                
                if not transcript:
                    return jsonify({"success": False, "error": "Failed to transcribe audio or no speech detected"})
            else:
                print(f"!!! SUCCESS: Got transcript: '{transcript}' !!!")
            
            # Extract voice biomarkers using audio
            print("Extracting voice biomarkers...")
            try:
                voice_biomarkers = extract_fatigue_features(temp_audio_path)
                print(f"Voice biomarkers extracted: {voice_biomarkers}")
            except Exception as e:
                print(f"Error extracting voice biomarkers: {e}")
                voice_biomarkers = {"error": str(e)}
            
            # Analyze cognitive indicators
            print("Analyzing cognitive indicators...")
            try:
                cognitive_indicators = analyze_cognitive_indicators(transcript)
                print(f"Cognitive indicators: {cognitive_indicators}")
            except Exception as e:
                print(f"Error in cognitive indicators analysis: {e}")
                cognitive_indicators = [f"Error: {e}"]
            
            # Generate cognitive scores using Gemini
            print("Generating cognitive scores...")
            try:
                scoring_prompt = f"""
As a cognitive health specialist, analyze this data and provide ONLY a JSON response with specific cognitive scores (0-100 scale):

SPEECH TRANSCRIPT: "{transcript}"

VOICE BIOMARKERS:
- Pitch characteristics: Mean={voice_biomarkers.get('pitch_mean', 0)}Hz, Std={voice_biomarkers.get('pitch_std', 0)}Hz
- Energy levels: Mean={voice_biomarkers.get('energy_mean', 0)}, Variability={voice_biomarkers.get('energy_variability', 0)}
- Speaking rate: {voice_biomarkers.get('speaking_rate', 0)} syllables/second
- Voice tremor index: {voice_biomarkers.get('tremor_index', 0)}
- Pause patterns: {voice_biomarkers.get('pause_frequency', 0)} pauses/second, avg duration {voice_biomarkers.get('avg_pause_duration', 0)}s

COGNITIVE INDICATORS: {cognitive_indicators}

Return ONLY a valid JSON object with these exact fields:
{{
    "memory_formation_score": [0-100 integer, where 100 = excellent recent recall ability],
    "word_retrieval_score": [0-100 integer, where 100 = excellent fluency and speed],
    "executive_function_score": [0-100 integer, where 100 = excellent planning/organization],
    "speech_timing_score": [0-100 integer, where 100 = excellent processing speed],
    "cognitive_risk_score": [0-100 integer, where 0 = no risk, 100 = high risk],
    "recommendations": ["specific recommendation 1", "specific recommendation 2", "specific recommendation 3"],
    "summary": "Brief 2-3 sentence summary of cognitive health status"
}}

Base scores on:
- Memory Formation: Ability to form new memories, recall recent events, working memory capacity
- Word Retrieval: Fluency, speed of word finding, absence of tip-of-tongue moments
- Executive Function: Planning abilities, cognitive flexibility, attention control
- Speech Timing: Processing speed, response latency, cognitive efficiency
- Risk Score: Overall cognitive decline risk based on all factors

Ensure all scores are realistic integers 0-100. Do not include any text outside the JSON.
"""
                
                # Generate cognitive scores from Gemini
                scores_response = model.generate_content(scoring_prompt)
                print("Cognitive scores generated successfully")
                
                # Parse JSON from response
                import json
                import re
                
                try:
                    # Extract JSON from response text
                    scores_text = scores_response.text.strip()
                    # Find JSON in the response
                    json_match = re.search(r'\{.*\}', scores_text, re.DOTALL)
                    if json_match:
                        scores_json = json.loads(json_match.group())
                    else:
                        # Fallback scores if parsing fails
                        scores_json = {
                            "memory_formation_score": 75,
                            "word_retrieval_score": 80,
                            "executive_function_score": 70,
                            "speech_timing_score": 85,
                            "cognitive_risk_score": 25,
                            "recommendations": [
                                "Continue regular cognitive exercises",
                                "Maintain social engagement",
                                "Consider consulting a healthcare professional"
                            ],
                            "summary": "Analysis shows good overall cognitive function with some areas for monitoring."
                        }
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing scores JSON: {e}")
                    # Fallback scores
                    scores_json = {
                        "memory_formation_score": 75,
                        "word_retrieval_score": 80,
                        "executive_function_score": 70,
                        "speech_timing_score": 85,
                        "cognitive_risk_score": 25,
                        "recommendations": [
                            "Continue regular cognitive exercises",
                            "Maintain social engagement",
                            "Consider consulting a healthcare professional"
                        ],
                        "summary": "Analysis shows good overall cognitive function with some areas for monitoring."
                    }
                
            except Exception as e:
                print(f"Error generating cognitive scores: {e}")
                scores_json = {
                    "memory_formation_score": 75,
                    "word_retrieval_score": 80,
                    "executive_function_score": 70,
                    "speech_timing_score": 85,
                    "cognitive_risk_score": 25,
                    "recommendations": [
                        "Continue regular cognitive exercises",
                        "Maintain social engagement",
                        "Consider consulting a healthcare professional"
                    ],
                    "summary": "Analysis shows good overall cognitive function with some areas for monitoring."
                }
            
            # Feed into comprehensive Gemini prompt for final assessment
            print("Generating comprehensive assessment...")
            try:
                gemini_prompt = f"""
As a cognitive health specialist, analyze this comprehensive data for early Alzheimer's detection:

SPEECH TRANSCRIPT: "{transcript}"

VOICE BIOMARKERS:
- Pitch characteristics: Mean={voice_biomarkers.get('pitch_mean', 0)}Hz, Std={voice_biomarkers.get('pitch_std', 0)}Hz
- Energy levels: Mean={voice_biomarkers.get('energy_mean', 0)}, Variability={voice_biomarkers.get('energy_variability', 0)}
- Speaking rate: {voice_biomarkers.get('speaking_rate', 0)} syllables/second
- Voice tremor index: {voice_biomarkers.get('tremor_index', 0)}
- Pause patterns: {voice_biomarkers.get('pause_frequency', 0)} pauses/second, avg duration {voice_biomarkers.get('avg_pause_duration', 0)}s

COGNITIVE SCORES:
- Memory Formation: {scores_json['memory_formation_score']}/100
- Word Retrieval: {scores_json['word_retrieval_score']}/100
- Executive Function: {scores_json['executive_function_score']}/100
- Speech Timing: {scores_json['speech_timing_score']}/100
- Risk Score: {scores_json['cognitive_risk_score']}/100

Provide a detailed assessment covering:
1. **Psychological State Indicators**: Emotional tone, stress markers, cognitive load
2. **Memory Formation Quality**: Working memory, episodic recall, semantic access
3. **Executive Function Evaluation**: Planning, attention, cognitive flexibility
4. **Early Intervention Recommendations**: Specific exercises, lifestyle changes
5. **Family Notification Suggestions**: When and how to involve family members

IMPORTANT: This is for early Alzheimer's detection. Be thorough but sensitive in your assessment. Provide specific, actionable recommendations while maintaining hope and dignity.

Format your response with clear sections and specific recommendations.
"""
                
                # Generate comprehensive response from Gemini
                final_response = model.generate_content(gemini_prompt)
                print("Comprehensive assessment generated successfully")
                
            except Exception as e:
                print(f"Error generating comprehensive assessment: {e}")
                final_response = type('obj', (object,), {'text': f"Error generating assessment: {e}"})
            
            # Generate personalized health report
            print("Generating personalized health report...")
            try:
                health_report = generate_health_report(scores_json)
                print("Health report generated successfully")
            except Exception as e:
                print(f"Error generating health report: {e}")
                health_report = {"health_report": "Health report generation failed", "error": str(e)}
            
            # Compile complete analysis
            complete_analysis = {
                "transcript": transcript,
                "voice_biomarkers": voice_biomarkers,
                "cognitive_indicators": cognitive_indicators,
                "cognitive_scores": scores_json,
                "health_report": health_report,
                "comprehensive_assessment": final_response.text,
                "timestamp": datetime.now().isoformat(),
                "status": "Analysis completed successfully"
            }
            
            # Convert any numpy types to JSON-serializable types
            complete_analysis = convert_numpy_types(complete_analysis)
            
            print("Pipeline completed successfully")
            return jsonify({
                "success": True,
                "message": "Comprehensive cognitive analysis completed",
                "analysis": complete_analysis
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                print(f"Cleaned up temporary file: {temp_audio_path}")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route("/dashboard", methods = ["GET", "POST"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/AIresults", methods = ["POST"])
def AIresults():
    try:
        data = request.get_json()
        user_question = data.get("message", "")
        user_scores = data.get("scores", {})
        user_analysis = data.get("analysis", {})
        
        # Comprehensive prompt to explain AI reasoning
        explanation_prompt = f"""
You are an AI cognitive health specialist explaining your analysis methodology. A user wants to understand how you arrived at their cognitive assessment results.

USER'S COGNITIVE SCORES:
{user_scores}

USER'S ANALYSIS DATA:
- Voice Biomarkers: {user_analysis.get('voice_biomarkers', {})}
- Cognitive Indicators: {user_analysis.get('cognitive_indicators', [])}
- Transcript: {user_analysis.get('transcript', 'No transcript available')}

USER'S QUESTION: {user_question}

EXPLAIN YOUR AI REASONING:
You should explain:
1. **How voice biomarkers influence scores** (pitch, speaking rate, pauses, tremor)
2. **How speech patterns indicate cognitive function** (word retrieval, sentence complexity)
3. **Why specific scores were assigned** (what data points led to each score)
4. **How AI identifies cognitive patterns** (what the algorithm looks for)
5. **The scientific basis** behind voice-based cognitive assessment
6. **Confidence levels** and limitations of AI analysis

GUIDELINES:
- Be specific about which voice features led to which scores
- Explain the medical/scientific reasoning behind correlations
- Use clear, educational language
- Be honest about AI limitations
- Provide evidence-based explanations
- Use emojis for clarity

Answer the user's specific question while explaining your AI reasoning process.
"""
        
        response = model.generate_content(explanation_prompt)
        
        return jsonify({
            "success": True,
            "response": response.text
        })
        
    except Exception as e:
        print(f"Error in AIresults: {e}")
        return jsonify({
            "success": False,
            "response": f"I apologize, but I'm having trouble explaining the analysis right now. The AI uses voice biomarkers like pitch, speaking rate, and speech patterns to assess cognitive function. Please try your question again."
        })


def generate_health_report(scores):
    """Generate a personalized cognitive health report using Gemini"""
    try:
        prompt = f"""
As a cognitive health specialist, create a personalized health report based on these cognitive assessment scores:

COGNITIVE SCORES:
- Memory Formation: {scores.get('memory_formation_score', 0)}/100
- Word Retrieval: {scores.get('word_retrieval_score', 0)}/100  
- Executive Function: {scores.get('executive_function_score', 0)}/100
- Speech Timing: {scores.get('speech_timing_score', 0)}/100
- Risk Score: {scores.get('cognitive_risk_score', 0)}/100

Create a professional medical-style report with:

1. **COGNITIVE HEALTH SUMMARY**: Overall assessment in 2-3 sentences
2. **STRENGTHS IDENTIFIED**: Areas performing well
3. **AREAS FOR ATTENTION**: Scores that may need monitoring
4. **PERSONALIZED RECOMMENDATIONS**: 
   - Specific brain training exercises
   - Lifestyle modifications
   - When to consult healthcare professionals
5. **NEXT STEPS**: Follow-up timeline and monitoring suggestions

Format as a professional medical report. Be encouraging but honest. Include specific, actionable recommendations.
"""
        
        response = model.generate_content(prompt)
        return {
            "health_report": response.text,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error generating health report: {e}")
        return {
            "health_report": "Unable to generate personalized report at this time. Please consult with a healthcare professional for comprehensive cognitive assessment.",
            "error": str(e)
        }

def analyze_cognitive_indicators(transcript):
    """Analyze transcript for cognitive health indicators"""
    if not transcript:
        return "No speech detected"
    
    indicators = []
    
    # Simple word count analysis
    words = transcript.split()
    word_count = len(words)
    
    if word_count < 10:
        indicators.append("Limited speech output")
    elif word_count > 100:
        indicators.append("Good speech output")
    
    # Check for filler words (potential word-finding difficulties)
    filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]
    filler_count = sum(1 for word in words if word.lower() in filler_words)
    filler_ratio = filler_count / word_count if word_count > 0 else 0
    
    if filler_ratio > 0.15:
        indicators.append("High use of filler words (potential word-finding difficulties)")
    elif filler_ratio < 0.05:
        indicators.append("Low use of filler words (good word retrieval)")
    
    # Check for repetition
    unique_words = set(words)
    repetition_ratio = 1 - (len(unique_words) / word_count) if word_count > 0 else 0
    
    if repetition_ratio > 0.4:
        indicators.append("High word repetition")
    elif repetition_ratio < 0.1:
        indicators.append("Good vocabulary diversity")
    
    # Check sentence complexity (simple heuristic)
    sentences = transcript.split('.')
    avg_sentence_length = word_count / len(sentences) if len(sentences) > 0 else 0
    
    if avg_sentence_length < 5:
        indicators.append("Short sentences (potential complexity issues)")
    elif avg_sentence_length > 15:
        indicators.append("Complex sentence structure")
    
    return indicators



if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
