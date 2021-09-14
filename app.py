import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug import secure_filename
import pandas as pd
import Kratos as tp
from Kratos import TextRankProject as tp
uploaded_file_name = ""
UPLOAD_FOLDER = '/pycharm project/flask_try/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def handle(filename):
    data_thread = tp.read_csv(filename)
    # data = remove_duplicate(data_thread)
    thread_number, text = tp.get_needed_data(data_thread)

    count = 0
    g_count = 0
    redundent = thread_number[0]

    one_complete_thred = []

    for thread_iterator in thread_number:
        if thread_iterator == redundent:
            one_complete_thred.insert(count, text[g_count])
            count += 1
            g_count += 1

        else:
            # print(one_complete_thred)
            top_senteces = tp.summarization(one_complete_thred)
            context = tp.classify(top_senteces)
            # return top_senteces, context
            one_complete_thred.clear()
            count = 0
            one_complete_thred.insert(count, text[g_count])
            count += 1;
            g_count += 1
            redundent = thread_iterator

@app.route('/output')
def output():
    filename = request.args.get('filename')

    sentences, context = tp.text_rank_output(filename)

    # --------------------------------------------------------------

    return render_template('sentences.html', len=len(sentences), sentences=sentences, context=context)

    # --------------------------------------------------------------


@app.route('/')
def upload_file():
   return render_template("Index.html")


@app.route("/uploader", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('output',filename=file.filename))
    return redirect("output")
