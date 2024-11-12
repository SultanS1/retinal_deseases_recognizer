package com.life.retinal_desease_prediction_app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class MyModel(context: Context) {

    private val interpreter: Interpreter
    private val classLabels = getClassLabels()

    init {
        val model = loadModelFile(context, "your_model.tflite")
        interpreter = Interpreter(model)
        validateModelOutputSize()
    }

    /**
     * Classifies an image and returns the predicted class name.
     */
    fun classifyImage(bitmap: Bitmap): String {
        val inputImage = preprocessImage(bitmap, 224)
        val output = Array(1) { FloatArray(classLabels.size) }

        interpreter.run(inputImage, output)

        // Print all output values for debugging
        println("Model output:")
        output[0].forEachIndexed { index, confidence ->
            println("Class ${index + 1}: ${classLabels[index]} -> $confidence")
        }

        // Get the class with the highest confidence
        return getClassName(output[0])
    }
    /**
     * Preprocesses the image by resizing and normalizing it to the model's expected input format.
     */
    private fun preprocessImage(bitmap: Bitmap, imgSize: Int): Array<Array<Array<FloatArray>>> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgSize, imgSize, true)
        val input = Array(1) { Array(imgSize) { Array(imgSize) { FloatArray(3) } } }

        val mean = 127.5f
        val std = 127.5f

        for (y in 0 until imgSize) {
            for (x in 0 until imgSize) {
                val pixel = resizedBitmap.getPixel(x, y)
                input[0][y][x][0] = ((pixel shr 16 and 0xFF) - mean) / std
                input[0][y][x][1] = ((pixel shr 8 and 0xFF) - mean) / std
                input[0][y][x][2] = ((pixel and 0xFF) - mean) / std
            }
        }
        return input
    }

    /**
     * Returns the class label corresponding to the highest probability in the output.
     */
    private fun getClassName(output: FloatArray): String {
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
        return if (maxIndex in classLabels.indices) classLabels[maxIndex] else "Unknown"
    }

    /**
     * Logs each class's confidence score from the model output.
     */
    private fun logOutputConfidences(output: FloatArray) {
        output.forEachIndexed { index, confidence ->
            println("${classLabels[index]}: $confidence")
        }
    }

    /**
     * Validates the model's output tensor shape to ensure it matches the expected number of classes.
     */
    private fun validateModelOutputSize() {
        val outputShape = interpreter.getOutputTensor(0).shape()
        if (outputShape[1] != classLabels.size) {
            throw IllegalArgumentException(
                "Model output size (${outputShape[1]}) does not match the number of class labels (${classLabels.size})."
            )
        }
    }

    /**
     * Loads the TensorFlow Lite model from the assets folder.
     */
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        context.assets.openFd(modelPath).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).channel.use { fileChannel ->
                return fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        }
    }

    /**
     * Defines the list of class labels for model output.
     */
    private fun getClassLabels(): List<String> {
        return listOf(
            "Diabetic Retinopathy",                  // DR
            "Age-Related Macular Degeneration",      // ARMD
            "Macular Hole",                          // MH
            "Drusen",                                // DN
            "Myopia",                                // MYA
            "Branch Retinal Vein Occlusion",         // BRVO
            "Tilted Disc Syndrome",                  // TSLN
            "Epiretinal Membrane",                   // ERM
            "Lacquer Cracks",                        // LS
            "Macular Scar",                          // MS
            "Central Serous Retinopathy",            // CSR
            "Optic Disc Coloboma",                   // ODC
            "Central Retinal Vein Occlusion",        // CRVO
            "Toxoplasma Vasculitis",                 // TV
            "Arterial Hypertension",                 // AH
            "Optic Disc Pit",                        // ODP
            "Optic Disc Edema",                      // ODE
            "Staphyloma",                            // ST
            "Anterior Ischemic Optic Neuropathy",    // AION
            "Papilledema",                           // PT
            "Retinal Tear",                          // RT
            "Retinoschisis",                         // RS
            "Choroidal Rupture Syndrome",            // CRS
            "Edema",                                 // EDN
            "Retinal Pigment Epithelium Changes",    // RPEC
            "Macular Hemorrhage",                    // MHL
            "Retinitis Pigmentosa",                  // RP
            "Cotton Wool Spots",                     // CWS
            "Choroidal Blood",                       // CB
            "Optic Disc Pallor or Maculopathy",      // ODPM
            "Preretinal Hemorrhage",                 // PRH
            "Macular Neovascularization",            // MNF
            "Hemorrhage",                            // HR
            "Central Retinal Artery Occlusion",      // CRAO
            "Toxoplasmic Disease",                   // TD
            "Cystoid Macular Edema",                 // CME
            "Posterior Capsular Rupture",            // PTCR
            "Cystoid Fibrosis",                      // CF
            "Vitreous Hemorrhage",                   // VH
            "Macular Atrophy",                       // MCA
            "Vitreous Syneresis",                    // VS
            "Branch Retinal Artery Occlusion",       // BRAO
            "Plaque",                                // PLQ
            "Hyperpigmentation and Edema",           // HPED
            "Chorioretinal Lesion",
            "Placeholder Label"                      // Placeholder for the missing label // CL
        )
    }
}
