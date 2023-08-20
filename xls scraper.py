import requests as req

link = 'https://media.opap.gr/Excel_xlsx/1100/{}/kino_{}_{}.xlsx'

yearList = list(map('{:02}'.format, range(2003, 2023)))
monthList = list(map('{:02}'.format, range(1, 13)))

for year in yearList:
    for month in monthList:
        response = req.get(link.format(year, year, month), stream=True)
        if response.ok:
            with open('{}_{}.xlsx'.format(year, month),'wb') as f:
                f.write(response.content)
