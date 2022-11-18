from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/home')
def home2():
    return render_template('homepage.html')

    
@app.route('/error')
def error():
    return render_template('error.html')


@app.route('/aboutproject')
def aboutproject():
    return render_template('aboutproject.html')



@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/sourcecode')
def sourcecode():
    return render_template('sourcecode.html')

@app.route('/creator')
def creator():
    return render_template('creator.html')





@app.route('/prediction' , methods=['POST','GET'])
def prediction():
    data1 = int(float(request.form['a']))
    data2 = int(float(request.form['b']))
    data3 = int(float(request.form['c']))
    print(data1,data2,data3)
    arr = np.array([[data1, data2, data3]])
    output= model.predict(arr)


    def to_str(var):
     return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]
     
   
    # return render_template('prediction.html')

    if (output<4):
        return render_template('prediction.html',p=to_str(output), q=' No ')
    elif (output>4 & output<6):
        return render_template('prediction.html',p=to_str(output), q= ' Low ')
    elif (output>6 & output<8):
        return render_template('prediction.html',p=to_str(output), q=' Moderate ')
    elif (output>8 & output<9):
        return render_template('prediction.html',p=to_str(output), q=' High ')
    elif (output>9):
        return render_template('prediction.html',p=to_str(output), q=' Very Hogh ')
    
    else :
        return render_template('prediction.html',p=' N.A.', q= ' Undefined ')


if __name__ == "__main__":
    app.run(debug=True)















