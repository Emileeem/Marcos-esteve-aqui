from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import random
import joblib

# try:
model = joblib.load('./model.pkl')
# except Exception as e:
#     print("Ocorreu um erro ao carregar o modelo:", e)

bp = Blueprint('default', __name__, url_prefix='/')

def contar(string, caractere):
    count = 0
    for char in string:
        if char == caractere:
            count += 1
    return count

def contarLetras(string):
    count = 0
    for char in string:
        if char.isalpha():  # Correção: isalpha() em vez de isAlpha()
            count += 1
    return count

@bp.route('/', methods=["POST", "GET"])
def index():
    url = ''
    NumDots, UrlLength, AtSymbol, NumDash, NumPercent, NumQueryComponents, IpAddress, HttpsInHostname, PathLevel, PathLength, NumNumericChars = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if request.method == 'POST':
        url = request.form.get("url")
        UrlLength = len(url)
        AtSymbol = contar(url, "@")
        NumDash = contar(url, "-")
        NumPercent = contar(url, "%")
        NumQueryComponents = contar(url, "?")
        PathLevel = contar(url, "/")
        NumNumericChars = contarLetras(url) 
        if PathLevel > 1:
            PathLength = random.randint(1, 10)
    
    result = model.predict(NumDots, UrlLength, AtSymbol, NumDash, NumPercent, NumQueryComponents, IpAddress, HttpsInHostname, PathLevel, PathLength, NumNumericChars)

    if result:
        result = "Não é Phising"
    else:
        result = "Phising"
    return render_template('index.html', url=url, result = result)

