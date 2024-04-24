from flask_wtf import FlaskForm

from flask_wtf.file import FileAllowed
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired


class LoadPictureForm(FlaskForm):
    picture = FileField('Загрузка изображения', validators=[
        DataRequired('Specify file'),
        FileAllowed(['png', 'jpeg', 'jpg', 'bmp'], 'Только .png или .jpg форматы.')
    ])
    load_data_submit = SubmitField('Загрузить')


class DownloadForm(FlaskForm):
    download_data_submit = SubmitField('Скачать итоговый файл')

