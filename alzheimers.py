from flask import Flask
from deepgram import DeepgramClient
import numpy as np
import librosa 
import google.generativeai as genai
import os

app = Flask(__name__)
dg_client = DeepgramClient('c74782c7b1b9d342303804006520ac45d5975e9b')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def extract_fatigue_features(audio_path):
    """Extract voice features specifically relevant to fatigue detection"""
    y, sr = librosa.load(audio_path, sr=16000)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0_clean = f0[voiced_flag]
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

def transcribe_audio(audio_path):
    """Transcribe audio file to text using Deepgram"""
    try:
        with open(audio_path, 'rb') as audio:
            # Configure transcription options
            options = {
                "model": "nova-2",
                "smart_format": True,
                "punctuate": True,
                "diarize": False,
                "utterances": False
            }
            
            # Transcribe the audio
            response = dg_client.listen.live.v("1").transcribe_file(audio, options)
            
            # Extract the transcript
            transcript = ""
            for result in response:
                if result.channel.alternatives:
                    transcript += result.channel.alternatives[0].transcript + " "
            
            return transcript.strip()
            
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

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

def analyze_semantic_fluency_gemini(transcript, category="animals"):
    """Analyze semantic fluency using Gemini AI"""
    if not transcript:
        return None
    
    prompt = f"""
    Analyze this transcript for semantic fluency in the category: {category}.
    
    Transcript: "{transcript}"
    
    Evaluate:
    1. How many {category} are mentioned
    2. Semantic clustering (related words grouped together)
    3. Semantic switching (moving between subcategories)
    4. Repetitions within the category
    5. Overall fluency score
    
    Return a JSON object with:
    - total_category_words: count of unique {category} mentioned
    - repetition_in_category: count of repeated {category}
    - semantic_clusters: number of semantic clusters
    - semantic_switches: number of switches between clusters
    - fluency_score: 0-20 scale (higher = better fluency)
    - mentioned_words: list of {category} mentioned
    """
    
    try:
        response = model.generate_content(prompt)
        return {
            "category": category,
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

def analyze_cognitive_patterns_gemini(audio_path):
    """Comprehensive cognitive analysis using Gemini AI"""
    # Extract voice features
    voice_features = extract_fatigue_features(audio_path)
    
    # Transcribe audio
    transcript = transcribe_audio(audio_path)
    
    if not transcript:
        return {"error": "Failed to transcribe audio"}
    
    # Analyze with Gemini
    word_retrieval = analyze_word_retrieval_gemini(transcript)
    semantic_fluency = analyze_semantic_fluency_gemini(transcript, "animals")
    
    # Get comprehensive Gemini analysis
    comprehensive_prompt = f"""
    Analyze this transcript for cognitive decline indicators related to Alzheimer's disease.
    
    Transcript: "{transcript}"
    
    Provide a comprehensive analysis including:
    1. Language patterns (word finding difficulties, circumlocution)
    2. Semantic fluency and category naming
    3. Overall cognitive assessment
    4. Risk level (low/medium/high)
    5. Specific observations and recommendations
    
    Format as JSON with detailed analysis.
    """
    
    try:
        comprehensive_analysis = model.generate_content(comprehensive_prompt)
        
        return {
            "voice_features": voice_features,
            "transcript": transcript,
            "word_retrieval_analysis": word_retrieval,
            "semantic_fluency_analysis": semantic_fluency,
            "comprehensive_analysis": comprehensive_analysis.text,
            "cognitive_risk_indicators": {
                "high_tot_indicators": word_retrieval.get("tot_indicators", 0) > 5 if word_retrieval else False,
                "high_circumlocution": word_retrieval.get("circumlocution_count", 0) > 3 if word_retrieval else False,
                "low_fluency_score": semantic_fluency.get("fluency_score", 0) < 10 if semantic_fluency else False,
                "high_repetition": word_retrieval.get("word_repetition_count", 0) > 5 if word_retrieval else False
            }
        }
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        return {"error": f"Analysis failed: {e}"}

def measure_syntactic_complexity(transcript):
    """Measure syntactic complexity for cognitive assessment"""
    if not transcript:
        return None
    
    sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    if not sentences:
        return {"complexity_score": 0, "avg_sentence_length": 0, "clause_count": 0}
    
    # Count clauses (simple heuristic: count conjunctions and relative pronouns)
    clause_indicators = ['and', 'but', 'or', 'because', 'although', 'while', 'when', 'where', 'who', 'which', 'that']
    total_clauses = sum(1 for word in transcript.lower().split() if word in clause_indicators)
    
    # Calculate metrics
    avg_length = np.mean([len(s.split()) for s in sentences])
    complexity_score = (avg_length * 0.4) + (total_clauses * 0.6)
    
    return {
        "complexity_score": round(complexity_score, 2),
        "avg_sentence_length": round(avg_length, 2),
        "clause_count": total_clauses,
        "sentence_count": len(sentences)
    }


