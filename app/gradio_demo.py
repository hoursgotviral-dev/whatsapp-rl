import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
try:
    from env import make_env
    from env.environment import Observation  # Pydantic model
    ENV_OK = True
    print('🚀 ENV + Pydantic OK!')
except:
    ENV_OK = False
    print('❌ Import failed')

def safe_chat(task_id, message, history):
    try:
        if not ENV_OK:
            return history, '❌ Environment not ready'
        
        clean_task = task_id.split('-')[0]
        env = make_env(clean_task)
        obs = env.reset()  # Pydantic Observation
        
        # Access Pydantic attributes (NOT dict!)
        obs_stage = obs.stage
        obs_chat = obs.chat_history
        
        action = {'action_type': 'PROVIDE_INFO', 'message': f'Re: {message[:30]}'}
        obs2, reward, done, info = env.step(action)
        
        response = obs2.chat_history[-1] if obs2.chat_history else 'Thanks for inquiring!'
        history.append((message, response))
        
        metrics = f"Stage: {obs2.stage} | Reward: {reward:.2f} | Sentiment: {obs2.sentiment:.2f}"
        return history, metrics
        
    except Exception as e:
        import traceback
        return history, f'Error: {str(e)[:60]}\n{traceback.format_exc()[-200:]}'

print('🚀 WhatsApp RL Demo - Pydantic Edition')
print('Open: http://localhost:7860')

with gr.Blocks(title='WhatsApp RL Agent') as demo:
    gr.Markdown('# 🏢 WhatsApp Business RL Demo')
    
    task_dropdown = gr.Dropdown(['task1-easy', 'task2-medium', 'task3-hard'], 
                              value='task1-easy', label='Customer Difficulty')
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder='What products do you offer?', label='Customer')
    
    with gr.Row():
        send_btn = gr.Button('Send', variant='primary')
        clear_btn = gr.Button('Clear')
    
    metrics = gr.Textbox(label='Metrics', lines=2)
    
    send_btn.click(safe_chat, [task_dropdown, msg, chatbot], [chatbot, metrics])
    msg.submit(safe_chat, [task_dropdown, msg, chatbot], [chatbot, metrics])
    clear_btn.click(lambda: ([], ''), None, chatbot)

demo.launch(server_port=7860, share=False, show_error=True)
