def construct_prompt(sentence):
    #分割符号
    separator = "|"
    prompt_template = f"""{sentence} {separator}
Task Prompt: Extract all (aspect, opinion, category, sentiment) quadruples from the sentence. Use the output format: [A] X [O] Y [C] Z [S] W [SSEP], where [SSEP] is used to separate multiple quadruples. If the aspect or opinion is implicit, output "NULL". 
Few-shot Prompt: 
Explicit aspect and explicit opinion example. Input: The food is good. Output: [A] food [O] good [C] food quality [S] positive
Implicit aspect and explicit opinion example. Input: It tastes really good. Output: [A] NULL [O] good [C] food quality [S] positive
Explicit aspect and implicit opinion example. Input: The food was gone in minutes. Output: [A] food [O] NULL [C] food quality [S] positive
Implicit aspect and implicit opinion example. Input: No leftovers this time. Output: [A] NULL [O] NULL [C] food quality [S] positive"""

    return prompt_template



# #输入提示构造
# prompt = construct_prompt("This food is great and drinks are fine")
# print(prompt)





