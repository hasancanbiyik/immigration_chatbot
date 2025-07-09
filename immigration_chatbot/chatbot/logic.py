from sentence_transformers import SentenceTransformer, util
import random
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImmigrationChatbot:

    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1", threshold: float = 0.5):
        """
        Initialize the immigration chatbot with semantic search capabilities.
    
        Args:
            model_name: Name of the sentence transformer model to use
            threshold: Minimum similarity score for matching questions
        """
        self.model = SentenceTransformer(model_name)
        # Increase threshold slightly as matching will be more accurate
        self.threshold = threshold 
    
        # Load the full structured Q&A data from the JSON file
        self.all_qa_data = self._load_qa_pairs()
    
        # self.qa_pairs will now be {topic_key: [answer1, answer2, ...]}
        # This remains useful for retrieving answers once a topic is matched.
        self.qa_pairs = {topic: data["answers"] for topic, data in self.all_qa_data.items()}
    
        # Create two lists: one with all representative questions, and another
        # with the corresponding topic key for each question.
        self.corpus_questions = []
        self.corpus_topics = []
    
        for topic, data in self.all_qa_data.items():
            # Use the "questions" array for embeddings. Fallback to topic key if it doesn't exist.
            questions_to_embed = data.get("questions", [topic]) 
            for question in questions_to_embed:
                self.corpus_questions.append(question)
                self.corpus_topics.append(topic)
            
        # Now, create embeddings from the rich list of representative questions
        self.question_embeddings = self.model.encode(self.corpus_questions, convert_to_tensor=True)

        # Track conversation context
        self.conversation_history = []
        
    def _get_default_qa_data(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Provides hardcoded default Q&A data structure as a fallback.
        This is used if 'qa_data.json' is not found or cannot be read.
        """
        return {
            "h1b visa": {
                "answers": [
                    "The H-1B is a temporary work visa for specialized occupations requiring theoretical or technical expertise.",
                    "H-1B allows foreign workers in fields like tech, finance, healthcare, and engineering to work in the U.S.",
                    "It's valid for 3 years, renewable once to 6 years total, and typically requires a bachelor's degree or equivalent.",
                    "The H-1B has an annual cap of 65,000 visas plus 20,000 for advanced degree holders from U.S. universities."
                ],
                "follow_ups": [
                    "Would you like to know about H-1B application deadlines?",
                    "Are you interested in H-1B to green card transition options?",
                    "Do you have questions about H-1B salary requirements?"
                ]
            },
            "opt": {
                "answers": [
                    "OPT (Optional Practical Training) allows F-1 students to work in their field for up to 12 months after graduation.",
                    "Optional Practical Training provides eligible international students with temporary employment authorization.",
                    "You can extend OPT by 24 months if you're in a STEM field, for a total of 36 months.",
                    "Pre-completion OPT allows students to work part-time during studies and full-time during breaks."
                ],
                "follow_ups": [
                    "Would you like to know about OPT application deadlines?",
                    "Are you interested in STEM OPT extension requirements?",
                    "Do you have questions about working during OPT?"
                ]
            },
            "green card": {
                "answers": [
                    "A green card provides permanent resident status, allowing you to live and work permanently in the U.S.",
                    "There are several ways to get a green card: through family, employment, investment, or special programs.",
                    "Employment-based green cards have different categories (EB-1, EB-2, EB-3) based on skills and qualifications.",
                    "Green card holders can apply for U.S. citizenship after 5 years (or 3 years if married to a U.S. citizen)."
                ],
                "follow_ups": [
                    "Would you like to know about green card processing times?",
                    "Are you interested in employment-based green card categories?",
                    "Do you have questions about green card renewal?"
                ]
            },
            "f1 visa": {
                "answers": [
                    "The F-1 visa is for international students enrolled in academic programs at U.S. universities.",
                    "F-1 students can work on-campus and may be eligible for off-campus work through CPT or OPT.",
                    "You must maintain full-time enrollment and make satisfactory academic progress to keep F-1 status.",
                    "F-1 visa holders can transfer between schools but must update their SEVIS record."
                ],
                "follow_ups": [
                    "Would you like to know about F-1 work authorization options?",
                    "Are you interested in F-1 to H-1B transition?",
                    "Do you have questions about maintaining F-1 status?"
                ]
            },
            "cpt": {
                "answers": [
                    "CPT (Curricular Practical Training) allows F-1 students to work off-campus in jobs related to their studies.",
                    "CPT must be an integral part of your curriculum, like an internship or co-op program.",
                    "You can work part-time (20 hours or less) or full-time CPT, but full-time CPT affects OPT eligibility.",
                    "CPT requires authorization from your school's international student office before starting work."
                ],
                "follow_ups": [
                    "Would you like to know about CPT vs OPT differences?",
                    "Are you interested in CPT application requirements?",
                    "Do you have questions about CPT work restrictions?"
                ]
            }
        }

    def _load_qa_pairs(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load Q&A pairs and follow-up questions from 'qa_data.json' file.
        If the file is not found or an error occurs during loading, it falls back to default hardcoded data.
        """
        qa_file_path = os.path.join(os.path.dirname(__file__), 'qa_data.json') #
        if os.path.exists(qa_file_path): #
            try:
                with open(qa_file_path, 'r', encoding='utf-8') as f: #
                    logger.info(f"Loading QA data from {qa_file_path}") #
                    return json.load(f) #
            except json.JSONDecodeError as e: # Catch specific JSON decoding errors
                logger.error(f"Error decoding JSON from {qa_file_path}: {e}. Using hardcoded defaults.") #
                return self._get_default_qa_data() #
            except Exception as e: # Catch other potential file/IO errors
                logger.error(f"An unexpected error occurred reading {qa_file_path}: {e}. Using hardcoded defaults.") #
                return self._get_default_qa_data() #
        else:
            logger.warning(f"QA data file not found at {qa_file_path}. Using hardcoded defaults.") #
            return self._get_default_qa_data() #
    
    def _save_qa_pairs(self) -> None:
        """
        Save the current state of all_qa_data back to the JSON file.
        This is typically called after adding new Q&A pairs dynamically.
        """
        qa_file_path = os.path.join(os.path.dirname(__file__), 'qa_data.json') #
        try:
            with open(qa_file_path, 'w', encoding='utf-8') as f: #
                json.dump(self.all_qa_data, f, indent=4) # Save self.all_qa_data (the full structure)
            logger.info(f"Saved QA data to {qa_file_path}") #
        except Exception as e:
            logger.error(f"Error saving QA data to {qa_file_path}: {e}") #

    def get_follow_up_questions(self, topic: str) -> List[str]:
        """
        Generate relevant follow-up questions based on the topic.
        Retrieves follow-ups from the loaded 'all_qa_data' structure.
        """
        # Access self.all_qa_data (the full loaded JSON structure)
        # Use .get() with default empty dictionary to safely handle missing topics
        # Use .get() with default empty list to safely handle missing 'follow_ups' key
        return self.all_qa_data.get(topic, {}).get("follow_ups", []) #
    
    def answer_question(self, user_input: str) -> Tuple[str, Optional[str], List[str]]:
        """
        Process user input and return answer with topic and follow-ups.
        
        Args:
            user_input: User's question or input
            
        Returns:
            Tuple of (answer, matched_topic, follow_up_questions)
        """
        # Clean and preprocess input
        user_input = user_input.strip().lower() #
        
        # Log the interaction
        logger.info(f"Processing question: {user_input}") #
        
        # Get embedding for user input
        user_embedding = self.model.encode(user_input, convert_to_tensor=True) #
        
        # Calculate similarities
        scores = util.cos_sim(user_embedding, self.question_embeddings)[0] #
        
        # Find best match
        best_idx = scores.argmax().item() #
        best_score = scores[best_idx].item() #
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(), #
            "user_input": user_input, #
            "best_match": self.corpus_questions[best_idx] if best_score >= self.threshold else None, #
            "confidence": best_score #
        })
        
        if best_score >= self.threshold: #
            topic = self.corpus_topics[best_idx] #

            # get a random answer for that topic
            answer = random.choice(self.qa_pairs[topic]) #

            # get the follow up questions
            follow_ups = self.get_follow_up_questions(topic) #

            matched_question = self.corpus_questions[best_idx]
            logger.info(f"Matched topic: '{topic}' via question: '{matched_question}' (confidence: {best_score:.3f})")
            return answer, topic, follow_ups

            logger.info(f"Matched topic: {topic} (confidence: {best_score:.3f})") #
            return answer, topic, follow_ups #
        else:
            logger.info(f"No match found (best score: {best_score:.3f})") #
            return self._get_fallback_response(user_input), None, [] #
    
    def _get_fallback_response(self, user_input: str) -> str:
        """
        Generate a helpful fallback response when no match is found.
        """
        fallback_responses = [
            "I don't have specific information about that topic yet. Could you try rephrasing your question?",
            "That's not something I can help with right now. Try asking about H-1B visas, OPT, green cards, or F-1 visas.",
            "I'm still learning about that topic. For now, I can help with common immigration questions about work visas and student status.",
        ]
        
        # Check if user might be asking about a related topic
        if any(keyword in user_input for keyword in ["work", "job", "employment"]): #
            return "For work-related immigration questions, I can help with H-1B visas, OPT, or green cards. What specifically would you like to know?" #
        elif any(keyword in user_input for keyword in ["student", "study", "school"]): #
            return "For student immigration questions, I can help with F-1 visas, OPT, or CPT. What would you like to know?" #
        
        return random.choice(fallback_responses) #



    def add_qa_pair(self, topic: str, answers: List[str], follow_ups: Optional[List[str]] = None) -> None:
        """
        Add new Q&A pairs to the chatbot's knowledge base.
        Updates the in-memory data and saves it to 'qa_data.json'.
        
        Args:
            topic: The new topic key (e.g., "new visa type").
            answers: A list of possible answers for this topic.
            follow_ups: An optional list of follow-up questions for this topic.
        """
        # Ensure the topic exists in the full data structure
        if topic not in self.all_qa_data:
            self.all_qa_data[topic] = {"answers": [], "follow_ups": [], "questions": [topic]} # Add a default question
        
        # Add or extend answers and follow-ups
        self.all_qa_data[topic]["answers"].extend(answers)
        if follow_ups:
            self.all_qa_data[topic]["follow_ups"].extend(follow_ups)

        # Rebuild the entire corpus and embeddings to include the new data
        self.corpus_questions = []
        self.corpus_topics = []
        for t, data in self.all_qa_data.items():
            questions_to_embed = data.get("questions", [t])
            for question in questions_to_embed:
                self.corpus_questions.append(question)
                self.corpus_topics.append(t)
                
        self.question_embeddings = self.model.encode(self.corpus_questions, convert_to_tensor=True)
        
        # Update the qa_pairs dictionary as well
        self.qa_pairs = {topic: data["answers"] for topic, data in self.all_qa_data.items()}
        
        self._save_qa_pairs() # Persist the changes to the JSON file
        logger.info(f"Added/updated topic: {topic} and re-indexed all questions.")

    
    def get_conversation_stats(self) -> Dict:
        """
        Get statistics about the conversation history.
        """
        if not self.conversation_history: #
            return {"total_questions": 0, "successful_matches": 0, "match_rate": 0.0} #
        
        total = len(self.conversation_history) #
        successful = sum(1 for item in self.conversation_history if item["best_match"] is not None) #
        
        return {
            "total_questions": total, #
            "successful_matches": successful, #
            "match_rate": successful / total if total > 0 else 0.0, #
            "avg_confidence": sum(item["confidence"] for item in self.conversation_history) / total #
        }
    
    def save_conversation_history(self, filename: str) -> None:
        """
        Save conversation history to a specified file.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f: #
                json.dump(self.conversation_history, f, indent=2) #
            logger.info(f"Saved conversation history to {filename}") #
        except Exception as e:
            logger.error(f"Error saving conversation history to {filename}: {e}") #

# Example usage (for testing this file directly)
if __name__ == "__main__":
    # Initialize chatbot
    # Ensure qa_data.json is in the same directory for this to work correctly
    chatbot = ImmigrationChatbot(threshold=0.45) #
    
    # Example conversation
    test_questions = [
        "What is an H1B visa?",
        "Can I work after graduation with F1?",
        "How long can I stay on OPT?",
        "What's the difference between CPT and OPT?",
        "How do I get a green card?",
        "Can I change from F1 to H1B?",
        "Tell me about something else.", # Test fallback
        "What are the requirements for F-1?" # Test follow-ups
    ]
    
    print("=== Immigration Chatbot Demo ===\n") #
    
    for question in test_questions: #
        print(f"Q: {question}") #
        answer, topic, follow_ups = chatbot.answer_question(question) #
        print(f"A: {answer}") #
        
        if follow_ups: #
            print("Follow-up questions:") #
            for follow_up in follow_ups: #
                print(f"  • {follow_up}") #
        print("-" * 50) #
    
    # Show conversation stats
    stats = chatbot.get_conversation_stats() #
    print(f"\n=== Conversation Statistics ===") #
    print(f"Total questions: {stats['total_questions']}") #
    print(f"Successful matches: {stats['successful_matches']}") #
    print(f"Match rate: {stats['match_rate']:.1%}") #
    print(f"Average confidence: {stats['avg_confidence']:.3f}") #

    # Example of adding a new QA pair and saving it
    new_topic = "travel"
    new_answers = ["International travel on F-1 status requires a valid visa, I-20 with a recent travel signature, and a valid passport.",
                   "For H-1B holders, travel usually requires a valid H-1B visa stamp and a valid passport."]
    new_follow_ups = ["What is a travel signature?", "Can I travel while my green card application is pending?"]
    
    print(f"\n--- Adding new topic: {new_topic} ---")
    chatbot.add_qa_pair(new_topic, new_answers, new_follow_ups)
    print("New topic added and saved to qa_data.json.")

    # Test the new topic
    print("\nQ: Can I travel with an H1B visa?")
    answer, topic, follow_ups = chatbot.answer_question("Can I travel with an H1B visa?")
    print(f"A: {answer}")
    if follow_ups:
        print("Follow-up questions:")
        for follow_up in follow_ups:
            print(f"  • {follow_up}")
    print("-" * 50)
    
    # Save final conversation history
    chatbot.save_conversation_history("conversation_log.json")
