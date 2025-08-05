from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
import mysql.connector
import os
from datetime import datetime
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder path
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO('model/best.pt')

# Connect to MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Abhi@123",
    database="road_safety_db"
)
mycursor = mydb.cursor()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Detection route
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        # Save uploaded file
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run YOLO detection
        results = model(filepath)[0]
        image = cv2.imread(filepath)

        detection_data = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            label = f"{class_name} ({conf*100:.0f}%)"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            detection_data.append(label)

            # Save to DB
            sql = "INSERT INTO detections (image_path, label, confidence, timestamp) VALUES (%s, %s, %s, %s)"
            val = (filepath.replace("\\", "/"), class_name, round(conf * 100, 2), timestamp)
            mycursor.execute(sql, val)
            mydb.commit()

        # Save annotated image
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, image)

        # Relative path for displaying in HTML
        relative_image_path = os.path.relpath(annotated_path, os.path.join(os.path.dirname(__file__), 'static')).replace('\\', '/')
        return render_template('result.html', image_path=relative_image_path, detections=detection_data)
@app.route('/history')
def history():
    mycursor.execute("SELECT image_path, label, confidence, timestamp FROM detections ORDER BY id DESC")
    rows = mycursor.fetchall()
    return render_template('history.html', detections=rows)
@app.route('/clear_history', methods=['POST'])
def clear_history():
    mycursor.execute("DELETE FROM detections")
    mydb.commit()
    return render_template('history.html', detections=[])

@app.route('/export_csv')
def export_csv():
    import csv
    from flask import Response

    mycursor.execute("SELECT image_path, label, confidence, timestamp FROM detections ORDER BY id DESC")
    rows = mycursor.fetchall()

    # Create CSV response
    def generate():
        yield 'image_path,label,confidence,timestamp\n'
        for row in rows:
            yield ','.join(str(item) for item in row) + '\n'

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=detections.csv"})
from fpdf import FPDF

@app.route('/export_pdf')
def export_pdf():
    mycursor.execute("SELECT image_path, label, confidence, timestamp FROM detections ORDER BY id DESC")
    rows = mycursor.fetchall()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Road Surface Guard - Detection Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    # Table headers
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(80, 10, "Image", border=1, fill=True)
    pdf.cell(30, 10, "Label", border=1, fill=True)
    pdf.cell(30, 10, "Confidence", border=1, fill=True)
    pdf.cell(50, 10, "Timestamp", border=1, fill=True)
    pdf.ln()

    # Table data
    for row in rows:
        image_name = os.path.basename(row[0])
        pdf.cell(80, 10, image_name, border=1)
        pdf.cell(30, 10, row[1], border=1)
        pdf.cell(30, 10, f"{row[2]}%", border=1)
        pdf.cell(50, 10, row[3], border=1)
        pdf.ln()

    # Output as downloadable file
    return pdf.output(dest='S').encode('latin-1'), 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename="detection_report.pdf"'
    } 
# Run the app
if __name__ == '__main__':
    app.run(debug=True)