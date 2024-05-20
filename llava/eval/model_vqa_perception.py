import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torchvision import transforms


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import sys


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"^^^^ {model_name}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    #FIXME:TEST
    #model.to('cuda')

    print("#################")
    print(model)
    print("#################")
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = json.load(open(args.question_file, "r"))
    
    reference = json.load(open(args.reference, "r"))
    
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    uniqueID_answer = {}
    
    #######################################################################################################
    # Traverse the whole dataset, and build a target dict based on the "id"
    # The "id" is X_Y_Z styleX stands for scene, Y stands for keyframe, Z stands for the question sequence of this specific keyframe
    # The key in the target dict is X_Y, and the value is another dict subDict
    # The key in the subDict is the question sequence id, and the value is the answer sequence
    #######################################################################################################
    
    #TODO:
    target_dict = {}
    prediction_dict = {}
    for line in tqdm(questions):
        idx = line["id"]
        qs = line["conversations"][0]["value"][8:] #去掉抬头<image>/n
        
        idx_split = idx.split("_")
        if len(idx_split) != 3:
            print("Error in spliting the id")
            sys.exit()
        
        keyframe_id = idx_split[0]+'_'+idx_split[1]
        question_sequence_id = int(idx_split[2])
        
        if keyframe_id not in target_dict:
            target_dict[keyframe_id] = {}
            prediction_dict[keyframe_id] = {}
        
        target_dict[keyframe_id][question_sequence_id] = qs
        prediction_dict[keyframe_id][question_sequence_id] = ""
        
    
    
    
    for line in tqdm(reference):
        idx = line["id"] #X_Y_Z -> sceneID_keyframeID_questionID
        
        idx_split = idx.split("_")
        if len(idx_split) != 3:
            print("Error in spliting the id")
            sys.exit()
        
        keyframe_id = idx_split[0]+'_'+idx_split[1]
        question_sequence_id = int(idx_split[2])
        
        answer = line["answer"]
        prediction_dict[keyframe_id][question_sequence_id] = answer
        
        
        
    
    # print len of target dict
    print(f"LEN of target dict: {len(target_dict)}")
    #print(f"LEN of one target dict element: {len(target_dict["b789de07180846cc972118ee6d1fb027_b0e6fd5561454b2789c853e5350557a8"])}")
    
    
    transformm = transforms.ToTensor()
    
    for line in tqdm(questions):
        
        idx = line["id"]
        
        idx_split = idx.split("_")
        keyframe_id, question_sequence_id = idx_split[0]+'_'+idx_split[1], int(idx_split[2])
        
        image_files_all = line["image"]
        qs = line["conversations"][0]["value"][8:] #去掉抬头<image>/n
        
        cur_prompt = qs
        
        perception_prompt = "There are some possible important objects around the ego car: "
        
        if question_sequence_id >= 1:
            perception_prompt += prediction_dict[keyframe_id][0]
            qs = perception_prompt + qs
        
        
        # if keyframe_id in target_dict:
        #     for i in range(question_sequence_id):
        #         if i==0:  #忽略本身就是感知第一问的问题
        #             break
        #         else:
        #             if i in target_dict[keyframe_id]:
        #                 if i==0:
        #                     perception_prompt += prediction_dict[keyframe_id][i]
        #             qs += perception_prompt
        
        
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
        
	
        print(f"$$$$$$$ Current Question is:::: {qs}")
        
        
        conv = conv_templates[args.conv_mode].copy()
        
        # For a question, according to the keyframe_id, if there are previous question in target_dict and related answer in prediction_dict
        # then use this QA as previous conversations as context and insert them before current conversations
    
        
        
        # previous_turn_number = 0
                    #     break
                    # if prediction_dict[keyframe_id][i] != "":
                    #     previous_turn_number += 1
                    #     conv.append_message(conv.roles[0], target_dict[keyframe_id][i])
                    #     conv.append_message(conv.roles[1], prediction_dict[keyframe_id][i])
        
        
        # print(f"######### [question_id] ########: {idx}")
        # print(f"######### [Current Question] ###: {qs}")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        
        
        
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        
        # img_all = []
        # img_size = None
        # for img_path in image_files_all:
        #     imag = Image.open(os.path.join(args.image_folder, img_path)).convert('RGB')
        #     img_size = imag.size
        #     # imag = transformm(imag)
        #     img_all.append(imag)
        
        
        # #TODO: we use image_aspect_ratio=="pad"
        # if True:    
        #     for i in range(len(img_all)):
        #         img_all[i] = expand2square(img_all[i], tuple(int(x*255) for x in image_processor.image_mean))
        #         img_all[i] = image_processor.preprocess(img_all[i], return_tensors='pt')['pixel_values'][0]
            
        # #FIXME:
        
        # images_tensor = torch.stack(img_all)
        # processor = None
        
        # for i in range(len(img_all)):
        #     img_all[i] = expand2square(img_all[i], tuple(int(x*255) for x in processor.image_mean))
        #     img_all[i] = processor.preprocess(img_all[i], return_tensors='pt')['pixel_values'][0]  
        
        image_file_list = image_files_all
        # we have image_processor
        image_folder = args.image_folder
        image_rgb = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file_list]
        
        image_rgb = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image_rgb]
        image_rgb = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'].half().cuda() for i in image_rgb]
        
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_rgb,
                #images=images_tensor.unsqueeze(0).half().cuda(),
                #image_sizes=[img_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=8192,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # prediction_dict[keyframe_id][question_sequence_id] = outputs
        
        print(f"######### Answer is:::: {outputs}")

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"id": idx,
                                   "question": cur_prompt,
                                   "answer": outputs
                                   #"answer_id": ans_id,
                                   #"model_id": model_name,
                                   #"metadata": {}
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./v1_1_val_nus_q_only_test_llama.json")
    parser.add_argument("--reference", type=str, default="./reference_ep3tp01.json") #TODO: utilize the 
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama_3") #llava_v1
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
