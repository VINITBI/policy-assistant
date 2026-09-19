from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, client_ip, get_current
from app.api.schemas import ChatRequest, ChatResponse, FeedbackRequest, MessageOut, SessionOut
from app.db.session import get_db
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/query", response_model=ChatResponse, summary="Ask a question about a policy")
def query(body: ChatRequest, request: Request, background: BackgroundTasks, current: CurrentUser = Depends(get_current),
          db: Session = Depends(get_db)):
    res = chat_service.ask(db, user=current.user, question=body.question, policy_id=body.policy_id,
                           session_id=body.session_id, ip=client_ip(request))
    ev = res.pop("_eval", None)
    if ev and ev["run"]:
        background.add_task(chat_service.self_evaluate, res["message_id"], ev["question"], ev["context"], ev["answer"])
    return res


@router.get("/sessions", response_model=list[SessionOut], summary="My previous conversations")
def sessions(limit: int = 50, current: CurrentUser = Depends(get_current), db: Session = Depends(get_db)):
    return chat_service.list_sessions(db, current.user, limit)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut], summary="Messages of a conversation")
def messages(session_id: str, current: CurrentUser = Depends(get_current), db: Session = Depends(get_db)):
    _, msgs = chat_service.get_messages(db, current.user, session_id)
    return msgs


@router.delete("/sessions/{session_id}", summary="Delete a conversation")
def delete_session(session_id: str, request: Request, current: CurrentUser = Depends(get_current),
                   db: Session = Depends(get_db)):
    chat_service.delete_session(db, current.user, session_id, client_ip(request))
    return {"message": "Conversation deleted."}


@router.post("/messages/{message_id}/feedback", summary="Rate an answer (thumbs up / down)")
def feedback(message_id: str, body: FeedbackRequest, request: Request, current: CurrentUser = Depends(get_current),
             db: Session = Depends(get_db)):
    chat_service.feedback(db, current.user, message_id, body.rating, body.comment, client_ip(request))
    return {"message": "Thanks for your feedback!"}
