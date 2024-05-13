from typing import List, Optional

import fire

from llama import Llama, Dialog
# from web_app_local import data#, docs_and_scores
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


#from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
#embeddings=HuggingFaceEmbeddings()

def main(
        ckpt_dir: str,
        tokenizer_path: str,
        chapter: str,
        data: str,
        source: str,
        #temperature: float = 0.6,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 6,
        max_gen_len: Optional[int] = None
        #max_gen_len: int = 9000
    ):
         
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        # print(max_seq_len)
        # print(max_batch_size)
        # vectorstore = FAISS.load_local((chapter), embeddings, allow_dangerous_deserialization=True)
    
        # docs_and_scores = vectorstore.similarity_search_with_score(data)
        # print(docs_and_scores[0])
        # print(f"DATA ---------------------------------> {data}")

        dialogs: List[Dialog] = [
            [
                # {"role": "system", "content": "Always answer precisely"},
                # {"role": "user", "content":"Who is Tim Surma?" },#
                {"role": "system",
                "content": source,#str(docs_and_scores[0]), #Tim Surma is a professor in Amsterdan with PhD in Statistics, 48 years old and dad of 5
                },
                {"role": "user", "content": data},#str(data)  "Who is Tim Surma?"
            ]
        ]
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")
        
            return result['generation']['content']


if __name__ == "__main__":
    fire.Fire(main)