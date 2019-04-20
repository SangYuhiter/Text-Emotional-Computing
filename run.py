# -*- coding: utf-8 -*-
"""
@File  : run.py
@Author: SangYu
@Date  : 2019/4/9 12:12
@Desc  : 
"""
from flask import Flask, render_template, request
from FastTextModel import sentence_input,sentence

app = Flask(__name__)

label_dict = {"__label__1": "正向", "__label__0": "中立", "__label__-1": "负向"}


@app.route("/", methods=["GET", "POST"])
@app.route("/input", methods=["GET", "POST"])
def input():
    if request.method == "POST":
        input_text = request.form["input_text"]
        temp_result = sentence_input(str(input_text))
        label_probs = []
        for label in temp_result:
            label_probs.append([label_dict[label[0]], label[1]])
        return render_template("input.html", input_text=input_text, label_probs=label_probs)
    return render_template("input.html", input_text="", label_probs=[])


if __name__ == '__main__':
    app.run(debug=True)
    # print(sentence_input("你真是个大好人"))