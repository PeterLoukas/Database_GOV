import json

Users_hours = {
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
}

temp_file = json.dumps(Users_hours, indent=3)

# Save the dictionary to a JSON file
with open("user_hours - hours_user.json", "w") as file:
    file.write(temp_file)
