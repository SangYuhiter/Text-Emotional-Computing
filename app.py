# -*- coding: utf-8 -*-
"""
@File  : app.py
@Author: SangYu
@Date  : 2019/4/9 12:12
@Desc  : 
"""
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
@app.route("/input", methods=["GET", "POST"])
def input():
    if request.method == "POST":
        question = request.form["question"]
        result = get_result(question)
        return render_template("input.html", result=result)
    return render_template("input.html", result="")


def get_result(question):
    return question + "sangyu"


if __name__ == '__main__':
    app.run(debug=True)
