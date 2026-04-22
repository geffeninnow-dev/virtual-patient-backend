from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, consultation

app = FastAPI(
    title="Virtual Patient Platform API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(consultation.router)


@app.get("/")
def root():
    return {"message": "Virtual Patient Platform API is running"}