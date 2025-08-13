# main.py

import os
import shutil
import tempfile
import zipfile

import PySimpleGUI as sg

from pole_inspector_tool import process_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

layout = [
    [
        sg.Text("Select a folder of pole images or a ZIP"),
        sg.Input(key="-IN-"),
        sg.FolderBrowse(),
        sg.FileBrowse(file_types=(("ZIP Files", "*.zip"),)),
    ],
    [sg.Checkbox("Separate into pass/fail folders", key="-PASSFAIL-", default=True)],
    [sg.Checkbox("Organize by pole number", key="-BYPOLE-", default=False)],
    [
        sg.Checkbox(
            "Check for low-quality (blurry or pixelated)",
            key="-LOWQ-",
            default=False,
        )
    ],
    [sg.ProgressBar(1, orientation="h", size=(40, 20), key="-PROG-")],
    [sg.Output(size=(60, 10))],
    [sg.Button("Run"), sg.Button("Exit")],
]

window = sg.Window("Pole Inspector", layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    if event == "Run":
        src = values["-IN-"]
        if not src or (not os.path.isdir(src) and not zipfile.is_zipfile(src)):
            print("Please select a valid folder or ZIP.")
            continue

        # unpack ZIP if needed
        temp_input = tempfile.TemporaryDirectory()
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src, "r") as z:
                z.extractall(temp_input.name)
            img_root = temp_input.name
        else:
            img_root = src

        # prepare outputs
        temp_output = tempfile.TemporaryDirectory()
        out_base = temp_output.name

        if values["-PASSFAIL-"]:
            os.makedirs(os.path.join(out_base, "pass"), exist_ok=True)
            os.makedirs(os.path.join(out_base, "fail"), exist_ok=True)

        if values["-LOWQ-"]:
            os.makedirs(os.path.join(out_base, "low_quality"), exist_ok=True)

        # collect images
        img_paths = []
        for root, _, files in os.walk(img_root):
            for f in files:
                if os.path.splitext(f.lower())[1] in IMAGE_EXTS:
                    img_paths.append(os.path.join(root, f))

        total = len(img_paths)
        pg = window["-PROG-"]
        pg.UpdateBar(0, total)
        print(f"Found {total} images, processing…")

        # process loop
        counts = {"pass": 0, "fail": 0, "lowq": 0}
        for idx, path in enumerate(img_paths, start=1):
            result = process_image(path, filename=os.path.basename(path))
            clarity_pass = result["clarity_pass"]
            pixel_pass = result["pixelation_pass"]
            status = result["status"]
            pole_id = result["filename"].split("_", 1)[0]

            # low-quality: either blurry or pixelated
            if values["-LOWQ-"] and (not clarity_pass or not pixel_pass):
                dst = os.path.join(out_base, "low_quality", os.path.basename(path))
                shutil.copy2(path, dst)
                counts["lowq"] += 1

            # pass/fail
            if values["-PASSFAIL-"]:
                folder = "pass" if status == "PASS" else "fail"
                dest_dir = os.path.join(out_base, folder)
                if values["-BYPOLE-"]:
                    dest_dir = os.path.join(dest_dir, pole_id)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))
                counts[folder] += 1

            pg.UpdateBar(idx)
            window.refresh()

        print("Done!")
        if values["-PASSFAIL-"]:
            print("Pass:", counts["pass"], "Fail:", counts["fail"])
        if values["-LOWQ-"]:
            print("Low-quality (blurry or pixelated):", counts["lowq"])
        print("Results in:", out_base)
        print("You can zip or copy this folder anywhere.")

window.close()
