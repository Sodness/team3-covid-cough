from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,EmailField
from wtforms.validators import DataRequired, Length, EqualTo, Email


class UserCreateForm(FlaskForm):
    username = StringField('아이디',validators=[DataRequired(), Length(min=3,max=20)])
    password1 = PasswordField('비밀번호',validators=[DataRequired(), EqualTo('password2','비밀번호가 다릅니다.'),Length(min=3,max=20)])
    password2 = PasswordField('비밀번호 확인',validators=[DataRequired()])
    email = EmailField('이메일',validators=[DataRequired(), Email()])

class UserLoginForm(FlaskForm):
    username = StringField('아이디', validators=[DataRequired(), Length(min=3, max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])