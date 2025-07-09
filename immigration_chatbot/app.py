import streamlit as st
from chatbot.logic import ImmigrationChatbot 
import time
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="Immigration Chatbot",
    page_icon="🧑‍⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chatbot (cached for performance)
@st.cache_resource
def load_chatbot():
    """Loads the ImmigrationChatbot model and returns an instance."""
    return ImmigrationChatbot(threshold=0.65)

chatbot = load_chatbot()

# --- Sidebar with helpful links and stats ---
with st.sidebar:
    st.header("🔗 Helpful Resources")
    
    # Official resources
    st.markdown("### Official Sites")
    st.markdown("- [USCIS Official Site](https://www.uscis.gov/)")
    st.markdown("- [OPT Information](https://www.uscis.gov/opt)")
    st.markdown("- [H-1B Information](https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations)")
    st.markdown("- [Green Card Info](https://www.uscis.gov/green-card)")
    st.markdown("- [F-1 Student Visa](https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/students-and-employment)")
    
    # Dynamic Quick Topics
    st.markdown("### 💡 Quick Topics")
    if hasattr(chatbot, 'all_qa_data') and chatbot.all_qa_data:
        for topic_key, data in chatbot.all_qa_data.items():
            display_topic = topic_key.replace('_', ' ').title()
            brief_desc = data["answers"][0].split('.')[0] + "..." if data["answers"] else "No description available."
            st.markdown(f"- **{display_topic}**: {brief_desc}")
    else:
        st.markdown("Topics loading or not available.")
    
    # Session stats
    if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
        st.markdown("### 📊 Stats")
        # Filter for assistant messages to count valid questions answered by the bot
        total_questions_answered_by_bot = sum(1 for item in st.session_state.chat_history if item.get('role') == 'assistant')
        
        positive_feedback = sum(1 for item in st.session_state.chat_history 
                                if item.get('role') == 'assistant' and item.get('feedback') == "👍 Yes")
        
        st.metric("Bot Responses", total_questions_answered_by_bot)
        if positive_feedback > 0:
            st.metric("Positive Feedback", f"{positive_feedback}/{total_questions_answered_by_bot}")
        else:
            st.metric("Positive Feedback", "N/A")


# --- App Title and Description ---
st.title("🧑‍⚖️ Immigration Chatbot")
st.markdown(
    """
    Ask questions about **H-1B visas**, **OPT**, **Green Cards**, **F-1 visas**, and other immigration topics. 
    
    💡 **Try asking**: "What is an H-1B visa?" or "How long can I work on OPT?"
    
    🔒 Your conversation stays private and only lives during this session.
    """
)

# --- Initialize session state ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'question_count' not in st.session_state:
    st.session_state.question_count = 0

if 'feedback_message' not in st.session_state:
    st.session_state.feedback_message = None

# --- Action buttons ---
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🧹 Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.question_count = 0
        st.session_state.feedback_message = None
        st.rerun()

with col2:
    if st.button("📊 Show Stats", type="secondary"):
        if st.session_state.chat_history:
            stats = {
                "total_messages": len(st.session_state.chat_history),
                "total_bot_responses": sum(1 for item in st.session_state.chat_history if item.get('role') == 'assistant'),
                "topics_covered": len(set(item["topic"] for item in st.session_state.chat_history 
                                        if item.get("topic"))),
                "positive_feedback": sum(1 for item in st.session_state.chat_history 
                                       if item.get('role') == 'assistant' and item.get("feedback") == "👍 Yes")
            }
            st.json(stats)
        else:
            st.info("No conversation yet! Ask a question to get started.")

# --- Welcome Message (only if chat is empty) ---
if not st.session_state.chat_history:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            """
            Hi! I'm here to help with US immigration questions. 
            
            I can help you with:
            - **H-1B visas** and work authorization
            - **OPT** and student work options  
            - **Green cards** and permanent residency
            - **F-1 visa** requirements and transitions
            - **CPT** for students
            
            What would you like to know about?
            """
        )

# --- Handle pending follow-up questions ---
user_input_from_button = st.session_state.get('pending_question')
if user_input_from_button:
    # Clear the pending question immediately
    del st.session_state.pending_question

# --- Chat Input ---
user_input_typed = st.chat_input("Ask your immigration question here...")

# Consolidate user input: prioritize button click over typed input
user_input = user_input_from_button if user_input_from_button else user_input_typed

# --- Process New Message ---
if user_input:
    st.session_state.question_count += 1 
    st.session_state.feedback_message = None  # Clear any previous feedback message
    
    # Add user message to chat history immediately
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    
    # Get bot response
    response, topic, follow_ups = chatbot.answer_question(user_input)
    
    # Add assistant message to chat history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'feedback': None,
        'topic': topic,
        'follow_ups': follow_ups  # Store follow-ups in chat history
    })

# --- Display Chat History ---
for i, chat_item in enumerate(st.session_state.chat_history):
    if chat_item['role'] == 'user':
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(chat_item['content'])
    
    elif chat_item['role'] == 'assistant':
        with st.chat_message("assistant", avatar="🤖"):
            # Show thinking animation only for the most recent response
            if i == len(st.session_state.chat_history) - 1 and user_input:
                with st.spinner("Thinking..."):
                    time.sleep(0.5)
            
            st.markdown(chat_item['content'])
            
            # Show follow-up questions if they exist
            if chat_item.get('follow_ups'):
                st.markdown("### 🤔 Related Questions")
                num_cols = 3 
                cols = st.columns(num_cols)
                for j, follow_up in enumerate(chat_item['follow_ups']):
                    with cols[j % num_cols]:
                        if st.button(follow_up, key=f"followup_{i}_{j}"):
                            st.session_state.pending_question = follow_up
                            st.rerun()
            
            # --- Feedback section ---
            st.markdown("---")
            
            # Show feedback message if it exists and this is the current response
            if (i == len(st.session_state.chat_history) - 1 and 
                st.session_state.feedback_message and 
                chat_item.get('feedback')):
                if chat_item['feedback'] == "👍 Yes":
                    st.success("Thanks for the feedback! 🙏")
                else:
                    st.info("Thanks for letting us know. We'll work on improving!")
            
            # Show feedback buttons if no feedback given yet
            if chat_item.get('feedback') is None:
                col1_fb, col2_fb, col3_fb = st.columns([1, 1, 2])
                
                with col1_fb:
                    if st.button("👍 Helpful", key=f"helpful_{i}"):
                        st.session_state.chat_history[i]['feedback'] = "👍 Yes"
                        st.session_state.feedback_message = "positive"
                        st.rerun()
                
                with col2_fb:
                    if st.button("👎 Not helpful", key=f"not_helpful_{i}"):
                        st.session_state.chat_history[i]['feedback'] = "👎 No"
                        st.session_state.feedback_message = "negative"
                        st.rerun()
            
            # Show feedback status if already given
            elif chat_item.get('feedback'):
                if chat_item['feedback'] == "👍 Yes":
                    st.success("You found this helpful ✓")
                else:
                    st.info("You marked this as not helpful")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        💡 This chatbot provides general information only. For official guidance, please consult USCIS or an immigration attorney.
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Export conversation feature ---
if st.session_state.chat_history:
    st.markdown("### 📄 Export Conversation")
    if st.button("📥 Download Chat History"):
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": sum(1 for item in st.session_state.chat_history if item.get('role') == 'assistant'),
            "conversation": st.session_state.chat_history
        }
        
        st.download_button(
            label="💾 Download JSON",
            data=json.dumps(chat_data, indent=2),
            file_name=f"immigration_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
