package com.example.camera_ai

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity(), View.OnClickListener {

    private val REQUEST_CODE_CAMERA_PERMISSION = 100
    private val REQUEST_IMAGE_CAPTURE = 1
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val cameraButton = findViewById<Button>(R.id.cameraButton)
        cameraButton.setOnClickListener(this)

        // Initialize the TFLite model in a try-catch block for error handling
        try {
            interpreter = Interpreter(loadModelFile("pop_3_5_24_yolov8n_float32.tflite"))
            Toast.makeText(this, "Model Loaded", Toast.LENGTH_LONG).show()
            println("Model loaded");
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show()
        }
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.cameraButton -> {
                // Check for camera permission
                if (checkCameraPermission()) {
                    openCamera()
                } else {
                    requestCameraPermission()
                }
            }
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            REQUEST_CODE_CAMERA_PERMISSION
        )
    }

    @SuppressLint("QueryPermissionsNeeded")
    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (intent.resolveActivity(packageManager) != null) {
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
        } else {
            Toast.makeText(this, "No camera app found", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd(modelFileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @SuppressLint("SetTextI18n")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            // Process the image using the TFLite model
            processCameraImage(imageBitmap)
        }
    }

    private fun processCameraImage(bitmap: Bitmap) {
        try {
            // Resize and preprocess the image for the model
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            // Run inference on the processed image
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 25200, 85), DataType.FLOAT32)
            interpreter.run(processedImage.buffer, outputBuffer.buffer.rewind())

            // Process the output for bounding boxes, etc.
            val outputArray = outputBuffer.floatArray
            parseYOLOOutput(outputArray)
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error during image processing", Toast.LENGTH_SHORT).show()
        }
    }

    private fun parseYOLOOutput(outputArray: FloatArray) {
        // Placeholder for post-processing logic (e.g., extract bounding boxes, labels, and scores)
        Toast.makeText(this, "Model inference complete", Toast.LENGTH_SHORT).show()

        // TODO: Implement actual parsing and display of bounding boxes, confidence, and class labels.
    }

    override fun onDestroy() {
        super.onDestroy()
        // Close interpreter when no longer needed
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }}