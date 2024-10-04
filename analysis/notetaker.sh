#!/bin/bash
# E1
# Accuracy vs. number of turns
# 7 program dataset
# Edge case user dataset


estring="e3"

max_dialog_turns=10
downsample_size=0 # full set
# downsample_size=1 # testing

### Backbone Reference ###
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy backbone

### SINGLE PROGRAMS ###
# GPT-4
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildAndDependentCareTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs EarlyHeadStartPrograms --estring $estring --downsample_size $downsample_size  --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs InfantToddlerPrograms --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs EarnedIncomeTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs HeadStart --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ComprehensiveAfterSchool --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker

## GPT-3.5-turbo ##
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildAndDependentCareTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs EarlyHeadStartPrograms --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs InfantToddlerPrograms --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs EarnedIncomeTaxCredit --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs HeadStart --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
python3 ./analysis/benefitsbot.py --max_dialog_turns $max_dialog_turns --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ComprehensiveAfterSchool --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker

## ALL PROGRAMS ###
GPT-4
python3 ./analysis/benefitsbot.py --max_dialog_turns 20 --chatbot_model_name gpt-4o-2024-05-13 --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildAndDependentCareTaxCredit EarlyHeadStartPrograms InfantToddlerPrograms ChildTaxCredit EarnedIncomeTaxCredit HeadStart ComprehensiveAfterSchool --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker
GPT-3.5-turbo
python3 ./analysis/benefitsbot.py --max_dialog_turns 20 --chatbot_model_name gpt-3-5-turbo --synthetic_user_model_name gpt-3-5-turbo --predict_every_turn True --programs ChildAndDependentCareTaxCredit EarlyHeadStartPrograms InfantToddlerPrograms ChildTaxCredit EarnedIncomeTaxCredit HeadStart ComprehensiveAfterSchool --estring $estring --downsample_size $downsample_size --chatbot_strategy notetaker