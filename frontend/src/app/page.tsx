"use client";

import Image from "next/image";
import { useState } from "react";
import Head from "next/head";
import { Camera } from "lucide-react";

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedEvents, setExtractedEvents] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };
  const handleCapture = async () => {
    // if (!imageFile) {
    //   alert("Please select an image file first.");
    //   return;
    // }
    document.getElementById("file-Input")?.click();
    console.log('clicked')
  };
  const processImage = async () => {
    if (!imageFile) {
      alert("Please select an image file first.");
      return;
    }
    setIsProcessing(true);
    setError(null);
    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await fetch("api/extract-events", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Failed to process image");
      }
      const data = await response.json();
      setExtractedEvents(data.events);
    } catch (error) {
      console.error("Error processing image:", error);
      setError("Failed to process image. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const addToCalendar = async (event: any) => {
    if (extractedEvents.length === 0) return;

    try {
      const response = await fetch("/api/add-to-calendar", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(event),
      });
      if (!response.ok) {
        throw new Error("Failed to add event to calendar");
      }
      alert("Event added to calendar successfully!");
      setImageFile(null);
      setImagePreview(null);
      setExtractedEvents([]);
    } catch (error) {
      console.error("Error adding event to calendar:", error);
      setError("Failed to add event to calendar. Please try again.");
    }
  };

  return (
    <>
      <div className="min-h-screen bg-gray-100">
        <Head>
          {/* <title>Photo to Calendar App</title> */}
          <meta
            name="description"
            content="Extract events from images and add them to your calendar"
          />
          <link rel="icon" href="/favicon.ico" />
        </Head>
       
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-center mb-8">Photo to Calendar</h1>

        <div className="bg-white rounded-lg shadow-md p-6 max-w-md mx-auto">
          <div className="space-y-6">
            {/* Image Upload */}
            <div className="flex flex-col items-center">
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              
              {imagePreview ? (
                <div className="relative w-full h-64 mb-4">
                  <Image 
                    src={imagePreview}
                    alt="Uploaded timetable"
                    layout="fill"
                    objectFit="contain"
                    className="rounded-md"
                  />
                </div>
              ) : (
                <div 
                  onClick={handleCapture}
                  className="w-full h-64 border-2 border-dashed border-gray-300 rounded-md flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50"
                >
                  <Camera size={48} className="text-gray-400 mb-2" />
                  <p className="text-sm text-gray-500">Click to take a photo or upload an image</p>
                </div>
              )}
              
              <div className="flex space-x-4 mt-4">
                <button
                  onClick={handleCapture}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  {imagePreview ? 'Take New Photo' : 'Take Photo'}
                </button>
                
                {imagePreview && (
                  <button
                    onClick={processImage}
                    disabled={isProcessing}
                    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-green-300"
                  >
                    {isProcessing ? 'Processing...' : 'Extract Events'}
                  </button>
                )}
              </div>

              {error && (
                <p className="text-red-500 mt-2">{error}</p>
              )}
            </div>

            {/* Extracted Events */}
            {extractedEvents.length > 0 && (
              <div className="mt-8">
                <h2 className="text-xl font-semibold mb-4">Extracted Events</h2>
                <div className="space-y-4">
                  {extractedEvents.map((event, index) => (
                    <div key={index} className="border rounded-md p-4">
                      <h3 className="font-medium">{event.title}</h3>
                      <p className="text-sm text-gray-600">
                        {new Date(event.start).toLocaleString()} - {new Date(event.end).toLocaleString()}
                      </p>
                      {event.location && (
                        <p className="text-sm text-gray-500">Location: {event.location}</p>
                      )}
                    </div>
                  ))}
                  
                  <button
                    onClick={addToCalendar}
                    className="w-full py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  >
                    Add to Calendar
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      </div>
    </>
  );
}
