
import os
import typer
from huggingface_hub import HfFileSystem, hf_hub_download
import torch
import gguf
from safetensors.torch import save_file
from tqdm import tqdm



app = typer.Typer()


def gguf_to_safetensors(gt, target_file):
	tensors = {}

	progress = tqdm(gt)
	for t in progress:
		progress.set_description(t.name)
		tensors[t.name] = torch.tensor(t.data)

	save_file(tensors, target_file, metadata={'format': 'pt'})


@app.command()
def main (repo_path: str, local_dir: str, download_only: bool = False):
	segs = repo_path.split('/')
	repo_id = '/'.join(segs[:2])

	fs = HfFileSystem()
	files = fs.ls(repo_path, detail=False)
	gguf_files = [file[len(repo_id) + 1:] for file in files if file.endswith('.gguf')]
	#print('gguf_files:', gguf_files)

	for file in gguf_files:
		gguf_path = os.path.join(local_dir, file)
		if not os.path.exists(gguf_path):
			print(f'Downloading {repo_id}/{file}...')
			gguf_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
		if download_only:
			continue

		reader = gguf.GGUFReader(gguf_path)
		print(f'Converting {gguf_path} to safetensors...')
		gguf_to_safetensors(reader.tensors, gguf_path.replace('.gguf', '.safetensors'))

	print('Done.')


if __name__ == "__main__":
	app()
