import json

# Sample JSON data provided
users_json = '''{
  "values": [

    [
      "",
      "",
      "Sun",
      "",
      "",
      "Mon",
      "",
      "",
      "Tue",
      "",
      "",
      "Wed",
      "",
      "",
      "Thu",
      "",
      "",
      "Fri",
      "",
      "",
      "Sat"
    ],
    [
      "",
      "",
      "start_time",
      "end_time",
      "duration",
      "start_time",
      "end_time",
      "duration",
      "start_time",
      "end_time",
      "duration",
      "start_time",
      "end_time",
      "duration",
      "start_time",
      "end_time",
      "duration",
      "start_time",
      "end_time",
      "duration",
      "start_time",
      "end_time",
      "duration"
    ],
    [],
    [
      "",
      "Anastasia Masadi",
      "-",
      "-",
      "-",
      "8:00:00 AM",
      "3:00:00 PM",
      "7:00:00",
      "8:00:00 AM",
      "3:00:00 PM",
      "7:00:00",
      "8:00:00 AM",
      "3:00:00 PM",
      "7:00:00",
      "8:00:00 AM",
      "3:00:00 PM",
      "7:00:00",
      "8:00:00 AM",
      "3:00:00 PM",
      "7:00:00",
      "-",
      "-",
      "-"
    ],
    [
      "",
      "Nikitas Tsinnas",
      "-",
      "-",
      "-",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "-",
      "-",
      "-"
    ],
    [
      "",
      "Giorgos Ntotsios",
      "-",
      "-",
      "-",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "-",
      "-",
      "-"
    ],
    [
      "",
      "Petros Loukas",
      "-",
      "-",
      "-",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "-",
      "-",
      "-"
    ],
    [
      "",
      "Irini Skapeti",
      "-",
      "-",
      "-",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "-",
      "-",
      "-"
    ],
    [
      "",
      "George Stamoulis",
      "-",
      "-",
      "-",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "10:00:00 AM",
      "7:00:00 PM",
      "9:00:00",
      "-",
      "-",
      "-"
    ],
    [
      "",
      "Evangelia Delligianni",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "-",
      "11:00:00 AM",
      "2:00:00 PM",
      "3:00:00",
      "12:00:00 PM",
      "2:00:00 PM",
      "2:00:00",
      "-",
      "-",
      "-"
    ]
  ]
}'''

employee_json = '''
       [{
        "Name": "Anastasia Masadi",
        "Birthday": "4.12",
        "Home/Delivery address": "Eresou 44, 10681, Athens",
        "Country": "Greece",
        "Mobile": "6976729027",
        "Emergency contact": "Christos Athanasopoulos",
        "Emergency contact phone number": "6974762051",
        "Private email": "",
        "Role": "Design Director & Product Owner",
        "Starting date": "",
        "Ending date": "",
        "work email": "anastasia@pixelunicorn.co",
        "One sentence about me": "",
        "Photo": ""
    },
    {
        "Name": "Nikitas Tsinnas",
        "Birthday": "8.06",
        "Home/Delivery address": "Grigoriou Kousidi 30, Zografou, Attica 15772",
        "Country": "Greece",
        "Mobile": "6947885111",
        "Emergency contact": "Konstantinos Tsinnas",
        "Emergency contact phone number": "6940538122",
        "Private email": "",
        "Role": "Junior Developer",
        "Starting date": "",
        "Ending date": "N/A",
        "work email": "nikitas@pixelunicorn.co",
        "One sentence about me": "Senior ECE NTUA student",
        "Photo": ""
    },
    {
        "Name": "Giorgos Ntotsios",
        "Birthday": "31.08",
        "Home/Delivery address": "Tsamadou 8, Exarcheia, Attica",
        "Country": "Greece",
        "Mobile": "6983362483",
        "Emergency contact": "Roxani Ntotsiou",
        "Emergency contact phone number": "6984759108",
        "Private email": "giorgos.ntotsios@gmail.com",
        "Role": "Front-end developer",
        "Starting date": "1.07",
        "Ending date": "31.08",
        "work email": "giorgos@pixelunicorn.co",
        "One sentence about me": "",
        "Photo": ""
    },
    {
        "Name": "Petros Loukas",
        "Birthday": "8.06",
        "Home/Delivery address": "Antwniou Papadima 2, Pallini, Attica 15351",
        "Country": "Greece",
        "Mobile": "6988672150",
        "Emergency contact": "Zampia Skaraki",
        "Emergency contact phone number": "6976680503",
        "Private email": "peterloukas6@gmail.com",
        "Role": "Data scientist",
        "Starting date": "1.07",
        "Ending date": "31.08",
        "work email": "petros@pixelunicorn.co",
        "One sentence about me": "",
        "Photo": ""
    },
    {
        "Name": "Irini Skapeti",
        "Birthday": "5.09",
        "Home/Delivery address": "Olympias 10, Zografou,Attica 15772",
        "Country": "Greece",
        "Mobile": "6907548369",
        "Emergency contact": "Lamprini Goula",
        "Emergency contact phone number": "6945432394",
        "Private email": "eiriniskap@gmail.com",
        "Role": "Full-stack developer",
        "Starting date": "",
        "Ending date": "",
        "work email": "irini@pixelunicorn.co",
        "One sentence about me": "",
        "Photo": ""
    },
    {
        "Name": "George Stamoulis",
        "Birthday": "3.09",
        "Home/Delivery address": "Chiou 25, 10438, Athens, Greece",
        "Country": "Greece",
        "Mobile": "",
        "Emergency contact": "",
        "Emergency contact phone number": "",
        "Private email": "",
        "Role": "Senior full-stack developer",
        "Starting date": "",
        "Ending date": "",
        "work email": "",
        "One sentence about me": "",
        "Photo": ""
    },
    {
        "Name": "Evangelia Delligianni",
        "Birthday": "15.08",
        "Home/Delivery address": "Paparrigopoulou 37, Chalandri, Attica 152 33",
        "Country": "Greece",
        "Mobile": "6974175612",
        "Emergency contact": "Spyros Tsavos",
        "Emergency contact phone number": "6986726625",
        "Private email": "",
        "Role": "",
        "Starting date": "",
        "Ending date": "",
        "work email": "",
        "One sentence about me": "",
        "Photo": ""
    }]
'''



# Load the JSON data
users_data = json.loads(users_json)
employee_data = json.loads(employee_json)

# Extract the working hours and additional details for each employee
working_hours = {}
weekday_names = users_data["values"][0][2::3]
for user_row in users_data["values"][3:]:
    name = user_row[1]
    working_hours[name] = {}
    for i, weekday in enumerate(weekday_names):
        start_time = user_row[2 + i * 3]
        end_time = user_row[3 + i * 3]
        duration = user_row[4 + i * 3]
        working_hours[name][weekday] = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        }

# Match the employee's name with the working hours and additional details
output = []
for employee in employee_data:
    name = employee["Name"]
    if name in working_hours:
        employee["Working Hours"] = working_hours[name]
        output.append(employee)

# Print the final output
print(json.dumps(output, indent=2))

