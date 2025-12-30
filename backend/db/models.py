from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    messages: Mapped[List["Message"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    content: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    chat: Mapped["Chat"] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(512))
    pickle_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(default=func.now())
    processed: Mapped[bool] = mapped_column(default=False)
    num_chunks: Mapped[int] = mapped_column(default=0)
    query_enabled: Mapped[bool] = mapped_column(default=True)

    @property
    def collection_name(self) -> str:
        prefix = "doc_"
        if self.id is None:
            return f"{prefix}pending"
        return f"{prefix}{self.id}"