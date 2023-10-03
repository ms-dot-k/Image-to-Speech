import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, GenerationConfig

class Im2Speech(nn.Module):
    def __init__(self, args, num_sp_unit):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(f"microsoft/{args.architecture}")
        
        self.model.config.vocab_size = num_sp_unit
        self.model.output = nn.Linear(768, num_sp_unit)
        # tie
        self.model.git.embeddings.word_embeddings.weight = self.model.output.weight

        self.mode = args.mode
        if self.mode == 'test':
            self.beam_size = args.beam_size
        self.generation_config = GenerationConfig(
            _from_model_config = True,
            bos_token_id = 101,
            eos_token_id = 102,
            pad_token_id = 0,
            )

    def forward(self, x, target_tensor, target_len, inference=False):
        if not inference:
            input_ids = target_tensor.clone()
            target_tensor[target_tensor == 0] = torch.tensor(-100).cuda()
            max_tgt_len = target_tensor.size(1)
            tgt_key_mask = self.generate_key_mask(target_len, max_tgt_len).cuda()
            output = self.model(input_ids=input_ids,
                            attention_mask=tgt_key_mask,
                            pixel_values=x,
                            labels=target_tensor)
            im_token_size = self.model.git.encoder.layer[0].attention.self.image_patch_tokens
            output.logits = output.logits[:, im_token_size:]
        else:
            if self.mode == 'test':
                output = self.model.generate(pixel_values=x, max_length=512, early_stopping=True, num_beams=self.beam_size, generation_config=self.generation_config)
            else:
                output = self.model.generate(pixel_values=x, max_length=512, early_stopping=True, generation_config=self.generation_config)

        return output
    
    def generate_key_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [1] * length[i]
            mask += [0] * (sz - length[i])
            masks += [torch.tensor(mask)]
        masks = torch.stack(masks, dim=0)
        return masks