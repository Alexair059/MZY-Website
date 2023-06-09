from flask import Flask, render_template
from markupsafe import escape

import os
import sys
from flask_sqlalchemy import SQLAlchemy

import click

from flask import request, url_for, redirect, flash

from werkzeug.security import generate_password_hash, check_password_hash

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from werkzeug.utils import secure_filename # 文件上传功能导入库

from FGSM.FGSM_Test import check # 导入FGSM检测模块

import datetime
import random # 基于添加随机字符的时间戳为上传图片生成唯一文件名

WIN = sys.platform.startswith('win')
if WIN:  # 如果是 Windows 系统，使用三个斜线
    prefix = 'sqlite:///'
else:  # 否则使用四个斜线
    prefix = 'sqlite:////'

UPLOAD_PATH = 'uploads'
ALLOWED_EXTENSIONS = ('jpg', 'jpeg', 'png') # 文件上传功能配置

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = prefix + os.path.join(app.root_path, 'data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 关闭对模型修改的监控
app.config['SECRET_KEY'] = 'dev'  # 等同于 app.secret_key = 'dev'
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000 # 文件上传功能配置

# 在扩展类实例化前加载配置
db = SQLAlchemy(app)

class User(db.Model, UserMixin):  # 表名将会是 user（自动生成，小写处理）
    id = db.Column(db.Integer, primary_key=True)  # 主键
    name = db.Column(db.String(20))  # 名字
    username = db.Column(db.String(20))  # 用户名
    password_hash = db.Column(db.String(128))  # 密码散列值

    def set_password(self, password):  # 用来设置密码的方法，接受密码作为参数
        self.password_hash = generate_password_hash(password)  # 将生成的密码保持到对应字段

    def validate_password(self, password):  # 用于验证密码的方法，接受密码作为参数
        return check_password_hash(self.password_hash, password)  # 返回布尔值


class Movie(db.Model):  # 表名将会是 movie
    id = db.Column(db.Integer, primary_key=True)  # 主键
    title = db.Column(db.String(60))  # 电影标题
    year = db.Column(db.String(4))  # 电影年份

class Pic(db.Model): # 为图站新创建的表
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(10))
    name = db.Column(db.String(20))
    desc = db.Column(db.String(128))
    uploader = db.Column(db.String(10))


login_manager = LoginManager(app)  # 实例化扩展类
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):  # 创建用户加载回调函数，接受用户 ID 作为参数
    user = User.query.get(int(user_id))  # 用 ID 作为 User 模型的主键查询对应的用户
    return user  # 返回用户对象

# 文件上传拓展名检查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/home')
def hello():
    return '<h1>Hello!</h1>'

@app.route('/user/<name>')
def user_page(name):
    return f'User: {escape(name)}'

@app.route('/MZY-Pic', methods=['GET', 'POST'])
def pic():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'check')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('未选择图片', 'check')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            suffix = filename.rsplit('.', 1)[1]
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
            picname = timestamp + '_' + random_str + '.' + suffix
            file.save(os.path.join(app.root_path, 'static', 'imgs', picname))
            flash('上传成功', 'check')
            pictitle = request.form.get('pictitle')  # 传入表单对应输入字段的 name 值
            picdes = request.form.get('picdes')
            picuploader = request.form.get('picloader')
            pic = Pic(path=picname, name=pictitle, desc=picdes, uploader=picuploader)
            db.session.add(pic)
            db.session.commit()
            return redirect(url_for('pic'))
    pics = Pic.query.all()
    return render_template('pic.html', pics = pics)

@app.route('/MZY-Pic/delete/<int:pic_id>', methods=['POST'])
def delpic(pic_id):
    pic = Pic.query.get_or_404(pic_id)
    os.remove(os.path.join(app.root_path, 'static', 'imgs', pic.path))
    db.session.delete(pic)
    db.session.commit()
    return redirect(url_for('pic'))

@app.route('/upload_files', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))
            flash('Upload success!!!', 'error')
    return render_template('upload.html')

@app.route('/FGSM', methods=['GET', 'POST'])
def FGSM():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'test')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file', 'test')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.root_path, 'test.png'))
            result = check(os.path.join(app.root_path, 'test.png'),
                           os.path.join(app.root_path, 'FGSM_img', 'std.png'), os.path.join(app.root_path, 'FGSM', 'MyNet.pt'))
            flash(result, 'test')
            os.remove(os.path.join(app.root_path, 'test.png'))
    return render_template('FGSM.html')

@app.context_processor
def inject_user():  # 函数名可以随意修改
    user = User.query.first()
    return dict(user=user)  # 需要返回字典，等同于 return {'user': user}

@app.route('/movie', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # 判断是否是 POST 请求
        if not current_user.is_authenticated:  # 如果当前用户未认证
            return redirect(url_for('index'))  # 重定向到主页
        # 获取表单数据
        title = request.form.get('title')  # 传入表单对应输入字段的 name 值
        year = request.form.get('year')
        # 验证数据
        if not title or not year or len(year) > 4 or len(title) > 60:
            flash('Invalid input.')  # 显示错误提示
            return redirect(url_for('index'))  # 重定向回主页
        # 保存表单数据到数据库
        movie = Movie(title=title, year=year)  # 创建记录
        db.session.add(movie)  # 添加到数据库会话
        db.session.commit()  # 提交数据库会话
        flash('Item created.')  # 显示成功创建的提示
        return redirect(url_for('index'))  # 重定向回主页

    movies = Movie.query.all()
    return render_template('index.html', movies=movies)

@app.route('/movie/edit/<int:movie_id>', methods=['GET', 'POST'])
@login_required  # 登录保护
def edit(movie_id):
    movie = Movie.query.get_or_404(movie_id)

    if request.method == 'POST':  # 处理编辑表单的提交请求
        title = request.form['title']
        year = request.form['year']

        if not title or not year or len(year) != 4 or len(title) > 60:
            flash('Invalid input.')
            return redirect(url_for('edit', movie_id=movie_id))  # 重定向回对应的编辑页面

        movie.title = title  # 更新标题
        movie.year = year  # 更新年份
        db.session.commit()  # 提交数据库会话
        flash('Item updated.')
        return redirect(url_for('index'))  # 重定向回主页

    return render_template('edit.html', movie=movie)  # 传入被编辑的电影记录

@app.route('/movie/delete/<int:movie_id>', methods=['POST'])  # 限定只接受 POST 请求
@login_required  # 登录保护
def delete(movie_id):
    movie = Movie.query.get_or_404(movie_id)  # 获取电影记录
    db.session.delete(movie)  # 删除对应的记录
    db.session.commit()  # 提交数据库会话
    flash('Item deleted.')
    return redirect(url_for('index'))  # 重定向回主页

@app.errorhandler(404)  # 传入要处理的错误代码
def page_not_found(e):  # 接受异常对象作为参数
    return render_template('404.html'), 404  # 返回模板和状态码

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash('Invalid input.')
            return redirect(url_for('login'))

        user = User.query.first()
        # 验证用户名和密码是否一致
        if username == user.username and user.validate_password(password):
            login_user(user)  # 登入用户
            flash('Login success.')
            return redirect(url_for('index'))  # 重定向到主页

        flash('Invalid username or password.')  # 如果验证失败，显示错误消息
        return redirect(url_for('login'))  # 重定向回登录页面

    return render_template('login.html')

@app.route('/logout')
@login_required  # 用于视图保护，后面会详细介绍
def logout():
    logout_user()  # 登出用户
    flash('Goodbye.')
    return redirect(url_for('index'))  # 重定向回首页

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        name = request.form['name']

        if not name or len(name) > 20:
            flash('Invalid input.')
            return redirect(url_for('settings'))

        current_user.name = name
        # current_user 会返回当前登录用户的数据库记录对象
        # 等同于下面的用法
        # user = User.query.first()
        # user.name = name
        db.session.commit()
        flash('Settings updated.')
        return redirect(url_for('index'))

    return render_template('settings.html')

@app.cli.command()
def forge():
    """Generate fake data."""

    # 全局的两个变量移动到这个函数内
    name = 'Alexair'
    movies = [
        {'title': 'My Neighbor Totoro', 'year': '1988'},
        {'title': 'Dead Poets Society', 'year': '1989'},
        {'title': 'A Perfect World', 'year': '1993'},
        {'title': 'Leon', 'year': '1994'},
        {'title': 'Mahjong', 'year': '1996'},
        {'title': 'Swallowtail Butterfly', 'year': '1996'},
        {'title': 'King of Comedy', 'year': '1999'},
        {'title': 'Devils on the Doorstep', 'year': '1999'},
        {'title': 'WALL-E', 'year': '2008'},
        {'title': 'The Pork of Music', 'year': '2012'},
    ]
    pics = [
        {'path': 'Snow.jpg',
         'name': '雪',
         'desc': 'Bing壁纸收藏1，一直想去这样的地方看看',
         'uploader': 'r0setta'},
        {'path': 'Bridge.jpg',
         'name': '桥',
         'desc': 'Bing壁纸收藏2，很喜欢这个角度',
         'uploader': 'r0setta'},
        {'path': 'Night.jpg',
         'name': '夜晚',
         'desc': 'Bing壁纸收藏3，在这里逛逛一定很棒',
         'uploader': 'r0setta'}
    ]

    user = User(name=name)
    db.session.add(user)
    for m in movies:
        movie = Movie(title=m['title'], year=m['year'])
        db.session.add(movie)

    for p in pics:
        pic = Pic(path=p['path'], name=p['name'], desc=p['desc'], uploader=p['uploader'])
        db.session.add(pic)

    db.session.commit()
    click.echo('Done.')

@app.cli.command()  # 注册为命令，可以传入 name 参数来自定义命令
@click.option('--drop', is_flag=True, help='Create after drop.')  # 设置选项
def initdb(drop):
    """Initialize the database."""
    if drop:  # 判断是否输入了选项
        db.drop_all()
    db.create_all()
    click.echo('Initialized database.')  # 输出提示信息

@app.cli.command()
@click.option('--username', prompt=True, help='The username used to login.')
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, help='The password used to login.')
def admin(username, password):
    """Create user."""

    user = User.query.first()
    if user is not None:
        click.echo('Updating user...')
        user.username = username
        user.set_password(password)  # 设置密码
    else:
        click.echo('Creating user...')
        user = User(username=username, name='Admin')
        user.set_password(password)  # 设置密码
        db.session.add(user)

    db.session.commit()  # 提交数据库会话
    click.echo('Done.')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)