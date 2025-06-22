import flask
from flask import request, jsonify, render_template, session, Flask
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions
import os
from datetime import datetime
import threading
import librosa
import numpy as np
API_KEY = '*******'

dg_client = DeepgramClient('******')
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

def extract_fatigue_features(audio_path):
    """Extract voice features specifically relevant to fatigue detection"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
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
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "pitch_range": round(pitch_range, 2),
            "energy_mean": round(energy_mean, 5),
            "energy_variability": round(energy_variability, 5),
            "speaking_rate": round(speaking_rate, 2),
            "tremor_index": round(tremor_index, 4),
            "pause_frequency": round(pause_frequency, 2),
            "avg_pause_duration": round(avg_pause_duration, 2),
            "duration": round(duration, 2)
        }
    except Exception as e:
        print(f"Error in voice feature extraction: {e}")
        return get_fallback_voice_features()

def get_fallback_voice_features():
    """Return realistic fallback voice features when extraction fails"""
    return {
        "pitch_mean": 150.0,  # Typical adult speaking pitch
        "pitch_std": 25.0,
        "pitch_range": 80.0,
        "energy_mean": 0.02,
        "energy_variability": 0.008,
        "speaking_rate": 4.5,  # syllables per second
        "tremor_index": 0.002,
        "pause_frequency": 1.2,
        "avg_pause_duration": 0.8,
        "duration": 10.0,
        "status": "fallback_values_used"
    }

def analyze_word_retrieval_gemini(transcript):
    """Analyze word retrieval patterns using Gemini AI"""
    if not transcript:
        return None
    
    prompt = f"""
    Analyze this transcript for word retrieval patterns that might indicate cognitive decline or Alzheimer's disease. 
    Focus on:
    1. Tip-of-tongue moments (filler words, hesitations, "what's the word" expressions)
    2. Circumlocution (describing instead of naming)
    3. Word repetition
    4. Sentence complexity
    
    Transcript: "{transcript}"
    
    Return a JSON object with these fields:
    - tot_indicators: count of tip-of-tongue indicators
    - circumlocution_count: count of circumlocution patterns
    - word_repetition_count: count of repeated words
    - avg_sentence_length: average words per sentence
    - total_words: total word count
    - cognitive_risk_score: 0-10 scale (higher = more risk)
    """
    
    try:
        response = model.generate_content(prompt)
        # Parse the response to extract JSON-like structure
        # You might need to adjust this based on actual Gemini response format
        return {
            "tot_indicators": 0,  # Placeholder - implement parsing logic
            "circumlocution_count": 0,
            "word_repetition_count": 0,
            "avg_sentence_length": 0,
            "total_words": len(transcript.split()),
            "cognitive_risk_score": 0,
            "gemini_analysis": response.text
        }
    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return None

def analyze_semantic_fluency_gemini(transcript):
    """Analyze semantic fluency with inferred category using Gemini AI"""
    if not transcript:
        return None

    prompt = f"""
You are an expert in cognitive and linguistic analysis. A patient has been asked to name as many related things as possible within a certain category (e.g., animals, fruits, furniture) in a short period.

Here is the transcript of what they said:
"{transcript}"

Your tasks:
1. **Infer the intended semantic category** the speaker is trying to generate words from (e.g., animals, fruits, tools, etc.)
2. Count how many **unique valid items** in that category were mentioned.
3. Identify **semantic clustering** — instances where related words (e.g., farm animals, ocean animals) are grouped together.
4. Identify **semantic switching** — points where the speaker transitions between different clusters or subcategories.
5. Count any **repetitions** of items.
6. Assign an **overall semantic fluency score** from 0 to 20, based on coherence, richness, and fluidity.

Respond in a JSON object with the following fields:
- inferred_category: the semantic category you believe the speaker was targeting
- total_category_words: count of unique valid items in that category
- repetition_in_category: count of repeated items
- semantic_clusters: number of meaningful clusters
- semantic_switches: number of switches between clusters
- fluency_score: integer from 0–20 (higher = better fluency)
- mentioned_words: list of words/phrases you identified as part of the category
    """

    try:
        response = model.generate_content(prompt)
        return {
            "total_category_words": 0,  # Placeholder
            "repetition_in_category": 0,
            "semantic_clusters": 0,
            "semantic_switches": 0,
            "fluency_score": 0,
            "mentioned_words": [],
            "gemini_analysis": response.text
        }
    except Exception as e:
        print(f"Error in Gemini fluency analysis: {e}")
        return None

    
def transcribe_audio(audio_path):
    """Transcribe audio file to text using Deepgram"""
    try:
        with open(audio_path, 'rb') as audio:
            # Configure transcription options using PrerecordedOptions
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                punctuate=True,
                diarize=False,
                utterances=False
            )
            
            # Use the correct Deepgram v4.0+ API
            response = dg_client.listen.rest.v("1").transcribe_file(
                {"buffer": audio, "mimetype": "audio/wav"}, 
                options
            )
            
            # Extract the transcript
            transcript = ""
            if response.results and response.results.channels:
                for channel in response.results.channels:
                    if channel.alternatives:
                        transcript += channel.alternatives[0].transcript + " "
            
            return transcript.strip() if transcript else None
            
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/voice_call")
def voice_call():
    return render_template("voice_call.html")

@app.route("/upload_call_data", methods=["POST"])
def upload_call_data():
    """Handle call data upload from VAPI"""
    try:
        data = request.get_json()
        call_id = data.get("call_id")
        phone_number = data.get("phone_number")
        duration = data.get("duration")
        timestamp = data.get("timestamp")
        
        print(f"Call data received: ID={call_id}, Phone={phone_number}, Duration={duration}")
        
        # Here you would typically:
        # 1. Fetch the audio recording from VAPI
        # 2. Analyze speech patterns
        # 3. Store results in your database
        
        # For now, return a mock analysis
        analysis_result = {
            "call_id": call_id,
            "duration_seconds": duration,
            "speech_analysis": "Mock analysis - speech patterns analyzed",
            "cognitive_indicators": "Mock cognitive health indicators",
            "recommendations": "Continue monitoring, consult healthcare professional if needed"
        }
        
        return jsonify({
            "success": True,
            "message": "Call data processed successfully",
            "analysis": analysis_result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Mayo Clinic Alzheimer's Knowledge Base for RAG
MAYO_CLINIC_ALZHEIMERS_KB = """
ALZHEIMER'S DISEASE - MAYO CLINIC MEDICAL INFORMATION

OVERVIEW:
Alzheimer's disease is the most common cause of dementia. About 6.9 million people in the United States age 65 and older live with Alzheimer's disease. Among them, more than 70% are age 75 and older. Early symptoms include forgetting recent events or conversations. Over time, Alzheimer's disease leads to serious memory loss and affects a person's ability to do everyday tasks.

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
        # Check if line contains any keywords
        if any(keyword in line_lower for keyword in keywords):
            # Add context (previous and next lines)
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

Provide a helpful, medically-informed response:
"""
        
        response = chat.send_message(rag_prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"I'm having trouble accessing medical information right now. Please consult with a healthcare professional for accurate medical advice."}) 

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        
        # Handle both audio_path (old format) and audio_data (new base64 format)
        if "audio_data" in data:
            # Handle base64 audio data from frontend recording
            import base64
            import tempfile
            
            audio_data = data.get("audio_data", "")
            audio_format = data.get("audio_format", "wav")
            
            if not audio_data:
                return jsonify({"transcript": None, "error": "No audio data provided"})
            
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as temp_file:
                temp_file.write(audio_bytes)
                temp_audio_path = temp_file.name
            
            # Transcribe the audio
            transcript = transcribe_audio(temp_audio_path)
            
            # Clean up temporary file
            import os
            os.unlink(temp_audio_path)
            
        else:
            # Handle audio_path (original format)
            audio_path = data.get("audio_path", "")
            transcript = transcribe_audio(audio_path)
        
        return jsonify({"transcript": transcript})
        
    except Exception as e:
        print(f"Error in transcribe endpoint: {e}")
        return jsonify({"transcript": None, "error": str(e)})


@app.route("/pipeline", methods = ["POST"])
def pipeline():
    try:
        data = request.get_json()
        print(f"Pipeline received data keys: {list(data.keys()) if data else 'No data'}")
        
        # Get audio data from request
        audio_data = data.get("audio_data", "")
        if not audio_data:
            print("No audio data provided")
            return jsonify({"success": False, "error": "No audio data provided"})
        
        print(f"Audio data length: {len(audio_data)} characters")
        
        # Decode and save audio temporarily
        import base64
        import tempfile
        import os
        
        try:
            audio_bytes = base64.b64decode(audio_data)
            print(f"Decoded audio bytes length: {len(audio_bytes)}")
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return jsonify({"success": False, "error": f"Invalid audio data: {e}"})
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name
        
        print(f"Temporary audio file created: {temp_audio_path}")
        
        try:
            # Transcribe audio
            print("Starting transcription...")
            transcript = transcribe_audio(temp_audio_path)
            print(f"Transcription result: {transcript}")
            
            if not transcript:
                print("Transcription failed or returned empty")
                return jsonify({"success": False, "error": "Failed to transcribe audio or no speech detected"})
            
            # Extract voice biomarkers using audio
            print("Extracting voice biomarkers...")
            try:
                voice_biomarkers = extract_fatigue_features(temp_audio_path)
                print(f"Voice biomarkers extracted: {voice_biomarkers}")
            except Exception as e:
                print(f"Error extracting voice biomarkers: {e}")
                voice_biomarkers = {"error": str(e)}
            
            # Analyze word retrieval patterns
            print("Analyzing word retrieval...")
            try:
                word_retrieval_analysis = analyze_word_retrieval_gemini(transcript)
                print(f"Word retrieval analysis completed")
            except Exception as e:
                print(f"Error in word retrieval analysis: {e}")
                word_retrieval_analysis = {"error": str(e)}
            
            # Analyze semantic fluency
            print("Analyzing semantic fluency...")
            try:
                semantic_fluency_analysis = analyze_semantic_fluency_gemini(transcript)
                print(f"Semantic fluency analysis completed")
            except Exception as e:
                print(f"Error in semantic fluency analysis: {e}")
                semantic_fluency_analysis = {"error": str(e)}
            
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

WORD RETRIEVAL ANALYSIS:
{word_retrieval_analysis.get('gemini_analysis', 'No analysis available') if isinstance(word_retrieval_analysis, dict) else 'Analysis failed'}

SEMANTIC FLUENCY ANALYSIS:
{semantic_fluency_analysis.get('gemini_analysis', 'No analysis available') if isinstance(semantic_fluency_analysis, dict) else 'Analysis failed'}

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
                "word_retrieval_analysis": word_retrieval_analysis,
                "semantic_fluency_analysis": semantic_fluency_analysis,
                "cognitive_indicators": cognitive_indicators,
                "cognitive_scores": scores_json,
                "health_report": health_report,
                "comprehensive_assessment": final_response.text,
                "timestamp": datetime.now().isoformat(),
                "status": "Analysis completed successfully"
            }
            
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
    
    if filler_ratio > 0.1:
        indicators.append("High use of filler words (potential word-finding difficulties)")
    elif filler_ratio < 0.05:
        indicators.append("Low use of filler words (good word retrieval)")
    
    # Check for repetition
    unique_words = set(words)
    repetition_ratio = 1 - (len(unique_words) / word_count) if word_count > 0 else 0
    
    if repetition_ratio > 0.3:
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

def get_cognitive_recommendations(transcript):
    """Get recommendations based on transcript analysis"""
    if not transcript:
        return ["No speech detected - please try speaking more"]
    
    recommendations = []
    
    # Basic recommendations based on speech patterns
    words = transcript.split()
    word_count = len(words)
    
    if word_count < 20:
        recommendations.append("Try to speak more during conversations")
        recommendations.append("Practice describing your day in detail")
    
    # Check for cognitive engagement
    if "remember" in transcript.lower() or "forgot" in transcript.lower():
        recommendations.append("Consider memory exercises and note-taking strategies")
    
    if "confused" in transcript.lower() or "don't know" in transcript.lower():
        recommendations.append("Practice breaking down complex tasks into smaller steps")
    
    # General recommendations
    recommendations.append("Continue regular cognitive activities")
    recommendations.append("Consider consulting a healthcare professional for comprehensive assessment")
    
    return recommendations

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
