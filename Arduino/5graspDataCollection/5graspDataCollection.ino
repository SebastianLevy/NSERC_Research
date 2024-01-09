// Define the pin where the sEMG sensor is connected
const int sEMGPin = A0;

// Define the baud rate for serial communication
const int baudRate = 9600;

void setup() {
  // Start the serial communication
  Serial.begin(baudRate);

  // Setup the sEMG pin as an input
  pinMode(sEMGPin, INPUT);
}

void loop() {
  // Read the value from the sEMG sensor
  int sEMGValue = analogRead(sEMGPin);

  // Print the sEMG value along with the grasp type
  // NOTE: The grasp type must be set manually for each collection session

  // For Cylindrical Grasp
  Serial.print("Cylindrical,");
  Serial.println(sEMGValue);

  // For Tip Grasp
  // Serial.print("Tip,");
  // Serial.println(sEMGValue);

  // For Hook Grasp
  // Serial.print("Hook,");
  // Serial.println(sEMGValue);

  // For Palmar Grasp
  // Serial.print("Palmar,");
  // Serial.println(sEMGValue);

  // For Lateral Grasp
  // Serial.print("Lateral,");
  // Serial.println(sEMGValue);

  // Wait for a short period before the next reading
  delay(10);
}
