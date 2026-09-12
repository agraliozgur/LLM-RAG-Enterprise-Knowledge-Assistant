import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chat_store import ChatStore


class ChatStoreTests(unittest.TestCase):
    def test_chat_is_autosaved_and_backed_up(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ChatStore(Path(temporary_directory), backup_limit=3)
            chat = store.create(["default"])
            store.append(chat["id"], {"role": "user", "content": "What does the contract say?"})
            store.append(chat["id"], {"role": "assistant", "answer": "Grounded answer", "sources": []})
            saved = store.get(chat["id"])
            self.assertEqual(saved["title"], "What does the contract say?")
            self.assertEqual(len(saved["messages"]), 2)
            self.assertTrue(list((Path(temporary_directory) / "data/cleaned_data/chat_backups").glob("*.json")))
            store.clear(chat["id"])
            self.assertEqual(store.get(chat["id"])["messages"], [])
