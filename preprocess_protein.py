import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import esm


class ESM(nn.Module):
    def __init__(self, device="cpu"):
        super(ESM, self).__init__()
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, sequence):
        if isinstance(sequence, tuple):
            sequences = [sequence]
        else:
            sequences = sequence

        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33]
        del results

        if len(sequences) == 1:
            return token_representations[0, 1:-1, :].cpu()
        else:
            return token_representations[:, 1:-1, :].cpu()


###########################################################
# main
###########################################################
if __name__ == "__main__":
    data_path = 'data'

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            print(f"GPU {i}: {total_mem / 1e9:.2f} GB total")
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")


    esm_model = ESM(device=device)

    dataset = 'Davis'
    maxlen = 1200

    file_path = f"data/{dataset}.txt"
    df = pd.read_csv(file_path, sep=" ")
    target_df = df.iloc[:, [1, 3]].drop_duplicates()

    key_targets = target_df.iloc[:, 0].tolist()
    seq_targets = target_df.iloc[:, 1].tolist()

    print(f"Total targets: {len(key_targets)}")

    sequences = list(zip(key_targets, seq_targets))

    save_dir = os.path.join(data_path, dataset, 'targets_ESM')
    os.makedirs(save_dir, exist_ok=True)

    print("Saving to directory:", save_dir)

    ###########################################################
    # Test a short sequence
    ###########################################################

    print("Testing with a short sequence...")
    test_seq = sequences[0]
    if len(test_seq[1]) > 100:
        test_seq = (test_seq[0], test_seq[1][:100])

    try:
        test_output = esm_model(test_seq)
        print(f"Test successful! Output shape: {test_output.shape}")
        del test_output
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)

    ###########################################################
    # Process all sequences
    ###########################################################

    success_count = 0

    for i, (key, seq) in enumerate(sequences):
        try:
            seq_len = len(seq)
            print(f"\nProcessing {i+1}/{len(sequences)}: {key}, length={seq_len}")

            if seq_len > maxlen:
                seq = seq[:maxlen]
                print(f"  Truncated to {maxlen}")

            sequence = (key, seq)

            output = esm_model(sequence)

            current_len = output.shape[0]

            if current_len > maxlen:
                output = output[:maxlen]
            elif current_len < maxlen:
                pad_size = maxlen - current_len
                output = F.pad(output, (0, 0, 0, pad_size), value=0)

            output = output.half()

            save_file = os.path.join(save_dir, f'{key}.pt')
            torch.save(output, save_file)

            print(f"  ✓ Saved (.pt, float16), shape: {output.shape}")

            success_count += 1

            del output

            if i % 5 == 4:
                torch.cuda.empty_cache()
                gc.collect()

        except torch.cuda.OutOfMemoryError:
            print(f"  ✗ OOM at sequence {i+1}, skipping...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    print(f"\nProcessing complete! Successfully processed {success_count}/{len(sequences)} targets.")
    print("All targets saved to directory:", save_dir)