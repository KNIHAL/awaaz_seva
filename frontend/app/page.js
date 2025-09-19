'use client'

import { useState, useEffect } from 'react'
import { Mic, MicOff, Send, Volume2, Loader2, Plus } from 'lucide-react'
import { useReactMediaRecorder } from 'react-media-recorder'

const API_BASE_URL = 'http://localhost:8000'

export default function Home() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState('')
  const [transcribedText, setTranscribedText] = useState('')
  const [messages, setMessages] = useState([])   // 🔹 Chat history
  const [selectedSession, setSelectedSession] = useState(null)

  const {
    status: recordingStatus,
    startRecording,
    stopRecording,
    mediaBlobUrl,
    clearBlobUrl
  } = useReactMediaRecorder({ audio: true })

  // Fetch chat history on load
  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/history`)
      const data = await res.json()
      setMessages(data.history || [])
    } catch (err) {
      console.error('Failed to fetch history', err)
    }
  }

  const startNewSession = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/new-session`, {
        method: 'POST',
      })
      const data = await res.json()
      setSelectedSession(data.session_id)
      setMessages([])
      setAnswer('')
      setTranscribedText('')
    } catch (err) {
      console.error('Failed to start session', err)
    }
  }

  // Text Question
  const handleTextQuestion = async () => {
    if (!question.trim()) return
    setIsLoading(true)
    setAnswer('')
    setAudioUrl('')
    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          language_preference: 'auto',
          include_audio: true,
          search_mode: 'auto'
        }),
      })
      const data = await response.json()
      setAnswer(data.answer)
      if (data.audio_url) setAudioUrl(`${API_BASE_URL}${data.audio_url}`)
      fetchHistory() // refresh history after sending
    } catch (err) {
      setAnswer('Sorry, something went wrong.')
    } finally {
      setIsLoading(false)
    }
  }

  // Voice Question
  const handleVoiceQuestion = async () => {
    if (!mediaBlobUrl) return
    setIsLoading(true)
    setAnswer('')
    setAudioUrl('')
    setTranscribedText('')
    try {
      const audioBlob = await fetch(mediaBlobUrl).then(r => r.blob())
      const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' })
      const formData = new FormData()
      formData.append('audio', audioFile)
      formData.append('language_preference', 'auto')
      formData.append('include_audio', 'true')
      formData.append('search_mode', 'auto')

      const response = await fetch(`${API_BASE_URL}/api/voice-ask`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setTranscribedText(data.transcription?.text || '')
      setAnswer(data.answer?.text || '')
      if (data.answer?.audio_url) setAudioUrl(`${API_BASE_URL}${data.answer.audio_url}`)
      fetchHistory()
    } catch (err) {
      setTranscribedText('🎤 (voice recorded, but backend is offline)')
      setAnswer('Sorry, voice processing failed. Please try again later.')
    } finally {
      setIsLoading(false)
      clearBlobUrl()
    }
  }

  const playAudio = () => {
    if (audioUrl) {
      new Audio(audioUrl).play().catch(e => console.error('Audio playback failed:', e))
    }
  }

  const toggleRecording = () => {
    if (recordingStatus === 'recording') stopRecording()
    else {
      startRecording()
      setTranscribedText('')
    }
  }

  return (
    <div className="flex h-screen bg-[#0f1740a] text-black-900">
      {/* Sidebar */}
      <aside className="w-64 bg-gray shadow-lg flex flex-col">
        <div className="p-4 flex items-center justify-between border-b">
          <span className="font-bold text-lg">Awaaz Seva</span>
          <button
            onClick={startNewSession}
            className="flex items-center bg-blue-500 text-white px-3 py-1 rounded-md hover:bg-blue-600"
          >
            <Plus size={16} className="mr-1" /> New
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {messages.length === 0 && (
            <p className="text-sm text-gray-500">No history yet</p>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`p-2 rounded-md text-sm cursor-pointer ${
                msg.type === 'user'
                  ? 'bg-blue-100 text-blue-800'
                  : 'bg-green-100 text-green-800'
              }`}
            >
              {msg.content.slice(0, 40)}...
            </div>
          ))}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <header className="p-4 flex justify-center border-b">
          <h1 className="text-4xl font-bold">Welcome to Awaaz Seva</h1>
        </header>

        {/* Chat Output */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {transcribedText && (
            <div className="bg-yellow-50 p-3 rounded-md border">
              <strong>You:</strong> {transcribedText}
            </div>
          )}
          {answer && (
            <div className="bg-white p-4 rounded-md shadow flex flex-col space-y-2">
              <div className="flex justify-between">
                <strong>Assistant</strong>
                {audioUrl && (
                  <button
                    onClick={playAudio}
                    className="flex items-center bg-green-500 text-white px-3 py-1 rounded-md hover:bg-green-600"
                  >
                    <Volume2 size={16} className="mr-1" /> Play
                  </button>
                )}
              </div>
              <p>{answer}</p>
            </div>
          )}
          {isLoading && (
            <div className="flex items-center text-blue-500">
              <Loader2 className="animate-spin mr-2" /> Processing...
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-3 border-t flex space-x-2">
          <form
            onSubmit={(e) => {
              e.preventDefault()
              handleTextQuestion()
            }}
            className="flex space-x-2 w-full"
           >
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Type your question here... (Hindi or English)"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
           />

            {/* Send Button */}
            <button
              type="submit"
              disabled={!question.trim() || isLoading}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 rounded-md flex items-center"
              >
              <Send size={16} />
            </button>

            {/* Mic Button */}
            <button
              type="button"
              onClick={toggleRecording}
              disabled={isLoading}
              className={`px-4 rounded-md flex items-center ${
                recordingStatus === 'recording'
                  ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
              >
              {recordingStatus === 'recording' ? <MicOff /> : <Mic />}
            </button>
          </form>

        </div>

        {/* Footer */}
        <footer className="p-2 text-center text-xs text-gray-500">
          Awaaz Seva project represented by Kumar Nihal to the IEEE
        </footer>
      </main>
    </div>
  )
}
