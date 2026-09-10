import subprocess
import time

files = [
# Add full file paths here in list format
]


total_counter = 0
batch_size = 40
saturation_multiplier = 4

while total_counter < len(files):
	batch_conter = 0
	command = [
		"C:\\Program Files\\GIMP 2\\bin\\gimp-2.10.exe", "-dfis",
	]

	for i in range(total_counter, total_counter+batch_size):
		if i >= len(files):
			break
		command.append("-b")
		command.append(f"(210-ColorSaturation-overwrite \"{files[i]}\" {saturation_multiplier})")
		total_counter += 1
		batch_conter += 1

	command.append("-b")
	command.append("(gimp-quit 0)")
	print(command)
	subprocess.call(command)

