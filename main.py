import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@app.get("/")
def read_root():
    return {"status": "Flood AI Backend is running"}

@app.post("/chat-analyze")
async def chat_analyze(
    message: str = Form(""),
    file: UploadFile = File(None)
):
    if not client:
        return {"reply": "❌ ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Environment Variables"}

    try:
        system_prompt = """
        คุณคือ 'Flood Safety AI' ผู้เชี่ยวชาญด้านประเมินภัยพิบัติน้ำท่วมและความปลอดภัย
        
        หลักการตอบ:
        1. ถ้ามีภาพ: ตรวจสอบความเสี่ยงน้ำท่วม/จุดอันตราย ระบุระดับความเสี่ยง และวิธีป้องกัน
        2. ถ้าเป็นข้อความ: ตอบคำถามเกี่ยวกับความปลอดภัยและการเตรียมรับมือน้ำท่วมอย่างกระชับ ชัดเจน
        """

        # 1. กรณีมีภาพแนบมา -> ใช้ Qwen 3.8 27B (โมเดล Vision ตัวเดียวที่ Groq เปิดอยู่)
        if file and file.filename:
            contents = await file.read()
            base64_image = base64.b64encode(contents).decode('utf-8')
            mime_type = file.content_type or "image/jpeg"
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            prompt_text = message.strip() if message.strip() else "ช่วยวิเคราะห์ภาพนี้ว่ามีปัญหาน้ำท่วมหรือความเสี่ยงอะไรไหม"

            messages_payload = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]

            completion = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=messages_payload,
                temperature=0.5,
                max_tokens=1024
            )

        # 2. กรณีส่งข้อความอย่างเดียว -> ใช้ GPT-OSS 120B (โมเดล Text หลักที่เสถียรที่สุดบน Groq)
        elif message and message.strip():
            messages_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message.strip()}
            ]

            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages_payload,
                temperature=0.7,
                max_tokens=1024
            )
            
        else:
            return {"reply": "กรุณาพิมพ์ข้อความหรือแนบรูปภาพส่งเข้ามาครับ"}

        if completion.choices and len(completion.choices) > 0:
            return {"reply": completion.choices[0].message.content}
        else:
            return {"reply": "AI ไม่ได้ตอบกลับข้อมูล กรุณาลองใหม่อีกครั้ง"}

    except Exception as e:
        return {"reply": f"⚠️ เกิดข้อผิดพลาดจาก AI: {str(e)}"}
        
