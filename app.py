from flask import Flask, send_file, jsonify, redirect, url_for, render_template
from model import generate_plot_and_metrics

app = Flask(__name__)

@app.route('/')
def all_links():
    links = ['plot','metrics','mlflow']
    return render_template('all_links.html', links=links)

@app.route('/plot')
def plot():
    buf, metrics = generate_plot_and_metrics()
    return send_file(buf, mimetype='image/png')

@app.route('/metrics')
def show_metrics():
    _, metrics = generate_plot_and_metrics()
    return jsonify(metrics)

@app.route('/mlflow')
def mlflow_ui():
    # This assumes MLflow is running on the default port 5000 on the host machine
    return redirect("http://127.0.0.1:5000")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)

