package com.life.retinal_desease_prediction_app

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.core.content.FileProvider
import com.life.retinal_desease_prediction_app.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.InputStream

class MainActivity : Activity() {

    private lateinit var model: MyModel
    private val PICK_IMAGE_REQUEST = 1
    private val REQUEST_IMAGE_CAPTURE = 2
    private var photoUri: Uri? = null
    private var _binding: ActivityMainBinding? = null
    private val binding get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        model = MyModel(this)

        setupButtons()
    }

    private fun setupButtons() {
        with(binding) {
            openGalleryButton.setOnClickListener { openGallery() }
        }
    }

    private fun openCamera() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val photoFile = createImageFile()

        photoFile?.let {
            photoUri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                it
            )
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
        } ?: run {
            showError("Error creating image file.")
        }
    }

    private fun openGallery() {
        val pickPhotoIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(pickPhotoIntent, PICK_IMAGE_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                PICK_IMAGE_REQUEST -> data?.data?.let { processImage(uriToBitmap(it)) }
                REQUEST_IMAGE_CAPTURE -> photoUri?.let { processImage(uriToBitmap(it)) }
            }
        }
    }

    private fun processImage(bitmap: Bitmap?) {
        bitmap?.let {
            binding.imageView.setImageBitmap(it)
            val result = model.classifyImage(it)
            displayResult(result)
        } ?: showError("Failed to load image.")
    }

    private fun displayResult(result: String) {
        binding.resultTextView.text = result
    }

    private fun showError(message: String) {
        binding.resultTextView.text = message
    }

    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun createImageFile(): File? {
        return try {
            val storageDir = cacheDir
            File.createTempFile("temp_image", ".jpg", storageDir)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        _binding = null // Avoid memory leaks
    }
}
