from datetime import datetime
import pandas as pd
import os

current_hour = datetime.now().strftime("%H:%M")
current_date = datetime.date(datetime.today())


spreadsheet_agenda = pd.read_excel(os.getcwd() + "\\agenda.xlsx")

description, sponsor, agenda_hour = [], [], []

for index, row in spreadsheet_agenda.iterrows():
    date = datetime.date(row['data'])
    hour = datetime.strptime(str(row['hora']), "%H:%M:%S")
    hour = datetime.time(hour).strftime("%H:%M")

    if date == current_date and hour > current_hour:
        description.append(row['descricao'])
        sponsor.append(row['responsavel'])
        agenda_hour.append(row['hora'])


def load_agenda():
    if len(description) > 0:
        return description, sponsor, agenda_hour
    else:
        return 'Nenhuma agenda encontrada'
