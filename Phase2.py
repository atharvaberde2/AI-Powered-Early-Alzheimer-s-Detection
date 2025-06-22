from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pgmpy.models import BayesianNetwork  # Leveraging your Bayesian expertise [3][4]

class ConversationOrchestrator:
    def __init__(self, user_baseline):
        self.user_baseline = user_baseline
        self.conversation_chains = {
            'daily_checkin': self.create_daily_checkin_chain(),
            'memory_assessment': self.create_memory_chain(),
            'executive_function': self.create_planning_chain(),
            'semantic_fluency': self.create_category_chain()
        }
        self.cognitive_model = self.build_bayesian_model()  # Using pgmpy [4]

    def build_bayesian_model(self):
        # Bayesian network for cognitive state inference
        model = BayesianNetwork([
            ('Memory_Recall', 'Cognitive_Score'),
            ('Planning_Complexity', 'Cognitive_Score'),
            ('Semantic_Diversity', 'Cognitive_Score')
        ])
        # Add CPDs based on clinical data [3]
        return model

    def create_daily_checkin_chain(self):
        return ConversationChain(
            memory=ConversationBufferMemory(),
            prompt=self._create_prompt("Describe your morning routine in detail"),
            output_key="emotional_coherence"
        )

    def create_memory_chain(self):
        return ConversationChain(
            memory=ConversationBufferMemory(),
            prompt=self._create_prompt("Retell yesterday's conversation about {topic}"),
            output_key="recall_accuracy"
        )

    def create_planning_chain(self):
        return ConversationChain(
            memory=ConversationBufferMemory(),
            prompt=self._create_prompt("Plan a meal with 3 courses using only {ingredients}"),
            output_key="plan_complexity"
        )

    def create_category_chain(self):
        return ConversationChain(
            memory=ConversationBufferMemory(),
            prompt=self._create_prompt("List as many {category} items as possible in 1 minute"),
            output_key="semantic_fluency"
        )

    def _create_prompt(self, template):
        return PromptTemplate(
            input_variables=["input"],
            template=template + "\nAssessment Focus: {cognitive_domain}"
        )

    def run_assessment(self, user_id):
        results = {}
        for domain, chain in self.conversation_chains.items():
            response = chain.run(cognitive_domain=domain)
            results[domain] = self._extract_metrics(response, domain)
        
        # Update Bayesian model with new evidence [4]
        self.cognitive_model.update_evidence(results)
        return self.cognitive_model.query('Cognitive_Score')

    def _extract_metrics(self, response, domain):
        # Domain-specific metric extraction
        metric_map = {
            'memory_assessment': self._calculate_recall_accuracy,
            'executive_function': self._score_plan_complexity,
            'semantic_fluency': self._count_semantic_diversity
        }
        return metric_map[domain](response)

    # Implement metric functions below...
    def _calculate_recall_accuracy(response):
        """Quantifies memory retention using semantic similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
    
        # Ground truth established during baseline session
        ground_truth = self.user_baseline['memory'][current_topic]  

        vectorizer = TfidfVectorizer().fit_transform([ground_truth, response])
        similarity = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
        return round(similarity * 100, 2)  # Convert to percentage
    def _score_plan_complexity(response):
        """Evaluates planning capability using structural analysis"""
        complexity_score = 0
    
        # Step-based scoring
        steps = [s.strip() for s in response.split('.') if s.strip()]
        complexity_score += min(len(steps) * 10, 40)  # Max 40 for steps
    
        # Conditional logic detection
        conditionals = sum(1 for step in steps if any(kw in step.lower() for kw in ['if', 'when', 'unless']))
        complexity_score += min(conditionals * 15, 30)  # Max 30 for conditionals
    
        # Resource allocation scoring
        resources = sum(1 for step in steps if any(kw in step.lower() for kw in ['quantity', 'amount', 'portion']))
        complexity_score += min(resources * 10, 30)  # Max 30 for resources
    
        return min(complexity_score, 100)  # Cap at 100%





class RiskAssessmentEngine:
    def calculate_cognitive_risk(self, current_data, baseline_data):
        decline_indicators = {
            'speech_timing_change': self.compare_timing_patterns(current_data, baseline_data),
            'vocabulary_decline': self.assess_vocabulary_changes(current_data, baseline_data),
            'memory_formation': self.evaluate_memory_performance(current_data),
            'executive_function': self.test_planning_abilities(current_data)
        }
        
        overall_risk = self.weighted_risk_calculation(decline_indicators)
        confidence_score = self.calculate_confidence(decline_indicators)
        
        return {
            'risk_score': overall_risk,
            'confidence': confidence_score,
            'decline_indicators': decline_indicators,
            'intervention_needed': overall_risk > 75
        }
    