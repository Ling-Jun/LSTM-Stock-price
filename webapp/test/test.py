from flask import Flask
app = Flask(__name__)


# To run in CLI with 'flask run', we have to name the this file 'app.py'.
# To run the app with 'python app.py', we just need to add "if __name__='__main__':" at the end.
@app.route('/')
def what():
    return 'Index Page, the page!'


@app.route('/hello')
def there():
    return 'Hello Page'


if __name__ == '__main__':
    app.run(debug=True)
