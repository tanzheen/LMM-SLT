#!/bin/sh

old_model_path="unsloth/Qwen2.5-3B-4bit"
new_model_path="unsloth/Qwen2.5-3B-4bit-pruned"
support_data="../../VLMEvalKit/raw_data/"
support_lang='de'
inherit_vocab_count="" # optional

# run pruning, check whether optional args are exists
if [ -z "$support_lang" -a -z "$inherit_vocab_count" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data"
elif [ -z "$inherit_vocab_count" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --support_lang $support_lang"
elif [ -z "$support_lang" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --inherit_vocab_count $inherit_vocab_count"
else
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --support_lang $support_lang --inherit_vocab_count $inherit_vocab_count"
fi
echo $cmd
$cmd

# run check the new tokenizer works as the same as old tokenizer in support data
cmd="python check.py --old_model_path=$old_model_path --new_model_path=$new_model_path --support_data=$support_data"
echo $cmd
$cmd
