from flask import Flask, request, session, redirect


def auth():
    if request.path.startswith('/static'):
        # 继续向后执行，不拦截
        return
    if request.path == '/login32':
        # 继续向后执行，不拦截
        return
    user = session.get('user')
    if user:
        # 继续向后执行，不拦截
        return
    return redirect('/login32')


def creat_app():
    app = Flask(__name__)
    app.secret_key = 'kjlaseru082340jdsldkqwil'

    from .views import account
    from .views import order
    app.register_blueprint(account.ac)
    app.register_blueprint(order.od)

    app.before_request(auth)
    return app