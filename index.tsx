import React, { useState, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type } from "@google/genai";
import { format } from "prettier/standalone";
import parserBabel from "prettier/plugins/babel";
import parserEstree from "prettier/plugins/estree";
import parserHtml from "prettier/plugins/html";
import parserPostcss from "prettier/plugins/postcss";
import parserTypescript from "prettier/plugins/typescript";

// --- Types & Interfaces ---
type Mode = "chat" | "code" | "image" | "video" | "audio" | "live";
type ChatModelType = "flash-lite" | "flash" | "pro" | "thinking";
type SpecializedMode = "default" | "research" | "shopping" | "study" | "image_gen";

interface Message {
  role: "user" | "model";
  content: string;
  type: "text" | "image" | "video" | "audio";
  imageData?: string; // base64
  videoUrl?: string; // base64 data url for persistence
  audioUrl?: string; // blob url
  timestamp: number;
  groundingMetadata?: any;
  isToolUse?: boolean;
}

interface ChatSession {
    id: string;
    title: string;
    messages: Message[];
    mode: 'chat' | 'code';
    updatedAt: number;
}

interface LibraryItem {
  id: string;
  type: 'image' | 'video';
  url: string; // Base64 Data URL
  prompt: string;
  timestamp: number;
}

// --- Constants ---
const MODEL_MAPPING: Record<ChatModelType, string> = {
  "flash-lite": "gemini-flash-lite-latest",
  "flash": "gemini-2.5-flash",
  "pro": "gemini-3-pro-preview",
  "thinking": "gemini-3-pro-preview",
};

const SUGGESTIONS = [
  { icon: "fa-solid fa-code", label: "Write Code", prompt: "Write a Python script to scrape a website." },
  { icon: "fa-solid fa-image", label: "Generate Image", prompt: "Generate a cyberpunk city with neon lights." },
  { icon: "fa-solid fa-video", label: "Create Video", prompt: "Create a video of a calm ocean sunset." },
  { icon: "fa-brands fa-google", label: "Search Web", prompt: "What are the latest tech news headlines today?" },
];

// --- Helper Functions ---

const fileToGenerativePart = async (file: File): Promise<{ inlineData: { data: string; mimeType: string } }> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = (reader.result as string).split(",")[1];
      resolve({
        inlineData: {
          data: base64String,
          mimeType: file.type,
        },
      });
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

const ensureApiKey = async () => {
  if ((window as any).aistudio && typeof (window as any).aistudio.hasSelectedApiKey === 'function') {
    const hasKey = await (window as any).aistudio.hasSelectedApiKey();
    if (!hasKey) {
      await (window as any).aistudio.openSelectKey();
    }
  }
};

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Shared Audio Context for TTS
let sharedAudioContext: AudioContext | null = null;

const playTTS = async (text: string, voiceName: string = 'Kore'): Promise<void> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-tts',
      contents: { parts: [{ text }] },
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName } } }
      }
    });

    const base64Data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Data) return;

    if (!sharedAudioContext) {
      sharedAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    }
    
    if (sharedAudioContext.state === 'suspended') {
      await sharedAudioContext.resume();
    }

    const bytes = base64ToUint8Array(base64Data);
    const int16 = new Int16Array(bytes.buffer);
    const buffer = sharedAudioContext.createBuffer(1, int16.length, 24000);
    const channelData = buffer.getChannelData(0);
    
    for (let i = 0; i < int16.length; i++) {
      channelData[i] = int16[i] / 32768.0;
    }

    const source = sharedAudioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(sharedAudioContext.destination);
    source.start();

  } catch (e) {
    console.error("TTS Error", e);
  }
};

// --- Tools Definition ---
const imageGenTool: FunctionDeclaration = {
  name: "generate_image",
  description: "Generate an image based on a detailed prompt.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: { type: Type.STRING, description: "The detailed prompt for the image." },
    },
    required: ["prompt"],
  },
};

const videoGenTool: FunctionDeclaration = {
  name: "generate_video",
  description: "Generate a video based on a detailed prompt.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: { type: Type.STRING, description: "The detailed prompt for the video." },
    },
    required: ["prompt"],
  },
};

// --- Components ---

function AppLogo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <div className={`${className} rounded-xl bg-gradient-to-br from-indigo-600 via-violet-600 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/20 relative overflow-hidden group border border-white/10 shrink-0`}>
      <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
      <span className="font-black text-white italic font-sans text-lg select-none">J</span>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 p-4 bg-slate-800/50 rounded-2xl w-fit animate-pulse mb-4 ml-4">
      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
    </div>
  );
}

function formatInline(text: string) {
    if (!text) return [];
    return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((chunk, j) => {
        if (chunk.startsWith('**') && chunk.endsWith('**')) return <strong key={j} className="text-white font-bold">{chunk.slice(2, -2)}</strong>;
        if (chunk.startsWith('`') && chunk.endsWith('`')) return <code key={j} className="bg-slate-700/50 px-1.5 py-0.5 rounded text-sm font-mono text-blue-300 border border-slate-700">{chunk.slice(1, -1)}</code>;
        return chunk;
    });
}

const FormattedText: React.FC<{ text: string }> = ({ text }) => {
   if (!text || !text.trim()) return null;
   return (
     <div className="text-slate-200 leading-relaxed text-[15px]">
       {text.split('\n').map((line, idx) => {
         let renderedLine: React.ReactNode = line;
         
         const trimmed = line.trim();
         
         if (line.startsWith('### ')) {
            renderedLine = <h3 className="text-lg font-bold text-slate-100 mt-5 mb-2 border-b border-slate-700/50 pb-1">{formatInline(line.slice(4))}</h3>;
         }
         else if (line.startsWith('## ')) {
            renderedLine = <h2 className="text-xl font-bold text-white mt-6 mb-3">{formatInline(line.slice(3))}</h2>;
         }
         else if (line.startsWith('# ')) {
            renderedLine = <h1 className="text-2xl font-bold text-white mt-8 mb-4">{formatInline(line.slice(2))}</h1>;
         }
         // List support
         else if (trimmed.startsWith('- ')) {
            renderedLine = (
              <div className="flex gap-3 ml-1 mb-1">
                <span className="text-slate-400 mt-1.5 w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0 block"></span>
                <span className="flex-1">{formatInline(trimmed.slice(2))}</span>
              </div>
            );
         }
         else if (trimmed.match(/^\d+\./)) {
            const match = trimmed.match(/^\d+\./);
            renderedLine = (
              <div className="flex gap-2 ml-1 mb-1">
                 <span className="text-slate-400 font-mono text-xs pt-1">{match![0]}</span>
                 <span className="flex-1">{formatInline(trimmed.replace(/^\d+\./, '').trim())}</span>
              </div>
            );
         }
         else {
            renderedLine = <p className="mb-2 min-h-[1.2em]">{formatInline(line)}</p>;
         }

         return <div key={idx}>{renderedLine}</div>
       })}
     </div>
   );
}

const CodeBlock: React.FC<{ code: string, language: string }> = ({ code, language }) => {
  const [formattedCode, setFormattedCode] = useState(code);

  useEffect(() => {
    let isMounted = true;
    const formatCode = async () => {
      try {
        let parser = null;
        let plugins: any[] = [];
        const lang = language.toLowerCase();

        if (['javascript', 'js', 'jsx', 'react', 'json'].includes(lang)) {
          parser = 'babel';
          plugins = [parserBabel, parserEstree];
        } else if (['typescript', 'ts', 'tsx'].includes(lang)) {
          parser = 'typescript';
          plugins = [parserTypescript, parserEstree];
        } else if (['html', 'xml', 'svg'].includes(lang)) {
          parser = 'html';
          plugins = [parserHtml];
        } else if (['css', 'scss', 'less'].includes(lang)) {
          parser = 'css';
          plugins = [parserPostcss];
        }

        if (parser) {
          const formatted = await format(code, {
            parser,
            plugins,
            printWidth: 80,
            tabWidth: 2,
            useTabs: false,
            singleQuote: false,
          });
          if (isMounted) setFormattedCode(formatted.trim());
        } else {
            if (isMounted) setFormattedCode(code);
        }
      } catch (e) {
        if (isMounted) setFormattedCode(code);
      }
    };

    formatCode();
    return () => { isMounted = false; };
  }, [code, language]);

  return (
    <div className="rounded-xl overflow-hidden border border-slate-700 bg-[#111] shadow-xl my-4 group">
      <div className="flex items-center justify-between px-4 py-2 bg-[#1a1a1a] border-b border-slate-800">
        <div className="flex items-center gap-2">
           <div className="flex gap-1.5">
             <div className="w-2.5 h-2.5 rounded-full bg-red-500/50"></div>
             <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50"></div>
             <div className="w-2.5 h-2.5 rounded-full bg-green-500/50"></div>
           </div>
           <span className="text-xs text-slate-500 font-mono uppercase ml-2">{language || 'code'}</span>
        </div>
        <button 
          onClick={() => navigator.clipboard.writeText(formattedCode)} 
          className="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1 opacity-0 group-hover:opacity-100 duration-200"
        >
          <i className="fa-regular fa-copy"></i> Copy
        </button>
      </div>
      <pre className="!m-0 !bg-transparent !border-0 text-sm overflow-x-auto p-4 text-slate-300 font-mono leading-relaxed">
        <code>{formattedCode}</code>
      </pre>
    </div>
  );
}

function SimpleMarkdown({ content }: { content: string }) {
  if (!content) return null;
  const parts = content.split(/(```[\s\S]*?```)/g);
  return (
    <div className="space-y-3">
      {parts.map((part, i) => {
        if (part.startsWith("```")) {
          const match = part.match(/```(\w*)\n([\s\S]*?)```/);
          if (match) {
            const [, lang, code] = match;
            return <CodeBlock key={i} code={code} language={lang} />;
          }
        }
        return <FormattedText key={i} text={part} />;
      })}
    </div>
  );
}

function FormattedContent({ message }: { message: Message }) {
  if (message.type === "image" && message.imageData) {
    return (
      <div className="space-y-2">
         {message.content && <p className="mb-2 whitespace-pre-wrap">{message.content}</p>}
         <img src={message.imageData} alt="Generated" className="rounded-xl shadow-lg max-w-full h-auto border border-slate-700/50" />
      </div>
    );
  }

  if (message.type === "video" && message.videoUrl) {
    return (
      <div className="space-y-2">
         {message.content && <p className="mb-2 whitespace-pre-wrap">{message.content}</p>}
         <video src={message.videoUrl} controls className="rounded-xl shadow-lg w-full max-w-md border border-slate-700/50" />
      </div>
    );
  }

  const groundingChunks = message.groundingMetadata?.groundingChunks;
  
  return (
    <div className="space-y-2">
      <SimpleMarkdown content={message.content} />
      
      {groundingChunks && groundingChunks.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700/30">
           <p className="text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-wider flex items-center gap-2">
             <i className="fa-solid fa-earth-americas"></i> Sources
           </p>
           <div className="flex flex-wrap gap-2">
             {groundingChunks.map((chunk: any, i: number) => {
               if (chunk.web?.uri) {
                 let hostname = "Source";
                 try {
                    hostname = new URL(chunk.web.uri).hostname;
                 } catch (e) {
                    // Safe fallback if URL is invalid
                 }
                 
                 return (
                   <a 
                     key={i} 
                     href={chunk.web.uri} 
                     target="_blank" 
                     rel="noreferrer" 
                     className="bg-slate-800/50 hover:bg-slate-700 border border-slate-700/50 hover:border-slate-600 px-3 py-1.5 rounded-full text-xs text-blue-400 hover:text-blue-300 truncate max-w-[200px] flex items-center gap-1.5 transition-all"
                   >
                     <img 
                        src={`https://www.google.com/s2/favicons?domain=${hostname}`} 
                        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        className="w-3 h-3 rounded-sm opacity-70" 
                        alt="" 
                     />
                     <span className="truncate">{chunk.web.title || hostname}</span>
                   </a>
                 );
               }
               return null;
             })}
           </div>
        </div>
      )}
    </div>
  );
}

const MessageBubble: React.FC<{ message: Message, selectedVoice?: string }> = ({ message, selectedVoice = "Kore" }) => {
  const isUser = message.role === "user";
  
  return (
    <div className={`flex gap-4 mb-8 ${isUser ? "flex-row-reverse" : "flex-row"} group animate-in fade-in slide-in-from-bottom-2 duration-300`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-lg mt-1 ${isUser ? "bg-slate-700" : "bg-gradient-to-br from-indigo-500 to-purple-600"}`}>
        {isUser ? <i className="fa-solid fa-user text-slate-300 text-xs"></i> : <AppLogo className="w-8 h-8 !text-xs" />}
      </div>
      
      <div className={`flex flex-col max-w-[85%] md:max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        <div className={`px-6 py-4 shadow-sm ${
          isUser 
            ? "bg-slate-700 text-white rounded-2xl rounded-tr-sm" 
            : "bg-slate-800/60 text-slate-200 border border-slate-700/50 rounded-2xl rounded-tl-sm backdrop-blur-sm shadow-black/20"
        }`}>
          <FormattedContent message={message} />
        </div>
        
        {/* Actions for Model Messages */}
        {!isUser && (
          <div className="flex gap-2 mt-2 ml-1 opacity-0 group-hover:opacity-100 transition-all duration-200">
            <button 
              onClick={() => playTTS(message.content, selectedVoice)}
              className="p-1.5 text-slate-500 hover:text-white hover:bg-slate-700/50 rounded-md transition-colors"
              title={`Read Aloud (${selectedVoice})`}
            >
              <i className="fa-solid fa-volume-high text-xs"></i>
            </button>
            <button 
              onClick={() => navigator.clipboard.writeText(message.content)}
              className="p-1.5 text-slate-500 hover:text-white hover:bg-slate-700/50 rounded-md transition-colors"
              title="Copy"
            >
               <i className="fa-regular fa-copy text-xs"></i>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Library View ---
const LibraryView = ({ items, onDelete }: { items: LibraryItem[], onDelete: (id: string) => void }) => {
  const [filter, setFilter] = useState<'all' | 'image' | 'video'>('all');

  const filteredItems = items.filter(item => filter === 'all' || item.type === filter);

  return (
    <div className="flex flex-col h-full bg-[#0f1117] p-6 overflow-hidden">
        <div className="flex items-center justify-between mb-8">
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
               <span className="bg-slate-800 p-2 rounded-xl text-indigo-500"><i className="fa-solid fa-photo-film"></i></span>
               Library
            </h1>
            <div className="flex gap-2 bg-slate-800/50 p-1 rounded-lg border border-slate-700/50">
               <button onClick={() => setFilter('all')} className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${filter === 'all' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}>All</button>
               <button onClick={() => setFilter('image')} className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${filter === 'image' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}>Images</button>
               <button onClick={() => setFilter('video')} className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${filter === 'video' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}>Videos</button>
            </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
           {filteredItems.length === 0 ? (
               <div className="flex flex-col items-center justify-center h-[60vh] text-slate-500">
                   <div className="w-20 h-20 bg-slate-800/50 rounded-full flex items-center justify-center mb-4 text-3xl opacity-50">
                       <i className="fa-regular fa-images"></i>
                   </div>
                   <p className="font-medium">Library is empty</p>
                   <p className="text-sm mt-1">Generated images and videos will appear here.</p>
               </div>
           ) : (
             <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 pb-10">
                {filteredItems.map(item => (
                   <div key={item.id} className="group relative bg-slate-800/30 border border-slate-700/50 rounded-xl overflow-hidden aspect-square hover:border-indigo-500/50 transition-all">
                       {item.type === 'image' ? (
                          <img src={item.url} alt={item.prompt} className="w-full h-full object-cover transition-transform group-hover:scale-105 duration-500" />
                       ) : (
                          <video src={item.url} className="w-full h-full object-cover" controls />
                       )}
                       
                       <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
                           <p className="text-white text-xs font-medium truncate mb-2">{item.prompt}</p>
                           <div className="flex gap-2">
                               <a href={item.url} download={`jeel-${item.type}-${item.id}`} className="flex-1 bg-white/10 hover:bg-white/20 text-white text-[10px] font-bold py-1.5 rounded text-center backdrop-blur-sm transition-colors">
                                   Download
                               </a>
                               <button onClick={(e) => { e.preventDefault(); onDelete(item.id); }} className="px-2 bg-red-500/20 hover:bg-red-500/40 text-red-400 rounded transition-colors">
                                   <i className="fa-solid fa-trash text-xs"></i>
                               </button>
                           </div>
                       </div>
                       
                       <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-sm text-white text-[10px] px-2 py-0.5 rounded-full border border-white/10">
                           <i className={`fa-solid ${item.type === 'image' ? 'fa-image' : 'fa-video'} mr-1`}></i>
                           {item.type === 'image' ? 'IMG' : 'VID'}
                       </div>
                   </div>
                ))}
             </div>
           )}
        </div>
    </div>
  );
};

// --- Login Screen ---
function LoginScreen({ onLogin }: { onLogin: (name: string) => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  
  const [isForgotPassword, setIsForgotPassword] = useState(false);
  const [resetEmail, setResetEmail] = useState("");
  const [resetStatus, setResetStatus] = useState<"idle" | "success">("idle");
  const [resetError, setResetError] = useState("");
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!email.trim()) { setError("Email is required"); return; }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) { setError("Invalid email format"); return; }
    if (!password) { setError("Password is required"); return; }
    if (password.length < 4) { setError("Wrong password, write right password"); return; }

    const name = email.split('@')[0];
    onLogin(name || email);
  };

  const handleSocialLogin = (provider: string) => {
    onLogin(`${provider} User`);
  };

  const handleForgotPasswordSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      setResetError("");
      if (!resetEmail.trim()) { setResetError("Email is required"); return; }
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(resetEmail)) { setResetError("Invalid email format"); return; }
      setResetStatus("success");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-600 to-blue-500 flex items-center justify-center p-4 relative overflow-hidden font-sans">
        {/* Decorative elements */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
           <div className="absolute top-[10%] left-[10%] w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
           <div className="absolute bottom-[20%] right-[10%] w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"></div>
        </div>

        <div className="w-full max-w-[420px] bg-white/95 backdrop-blur-xl p-8 rounded-[2rem] shadow-2xl relative z-10 animate-in fade-in zoom-in-95 duration-500 border border-white/50">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white p-3 rounded-2xl shadow-lg">
                <AppLogo className="w-12 h-12" />
            </div>

            {isForgotPassword ? (
                <div className="animate-in fade-in slide-in-from-right duration-300 mt-6">
                    <button 
                        onClick={() => { setIsForgotPassword(false); setResetStatus("idle"); setResetError(""); }}
                        className="text-slate-500 hover:text-slate-800 mb-6 flex items-center gap-2 text-sm transition-colors"
                    >
                        <i className="fa-solid fa-arrow-left"></i> Back to Login
                    </button>
                    <h1 className="text-2xl font-bold text-slate-800 mb-2">Reset Password</h1>
                    <p className="text-slate-500 text-sm mb-6">Enter your email and we'll send a link.</p>

                    {resetStatus === "success" ? (
                        <div className="text-center py-6">
                            <div className="w-14 h-14 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4 text-xl shadow-sm">
                                <i className="fa-solid fa-check"></i>
                            </div>
                            <h3 className="text-slate-800 font-semibold mb-2">Email Sent</h3>
                            <p className="text-slate-500 text-sm mb-6">Check <span className="font-semibold text-slate-700">{resetEmail}</span> for instructions.</p>
                            <button 
                                onClick={() => { setIsForgotPassword(false); setResetStatus("idle"); }}
                                className="w-full bg-slate-900 hover:bg-slate-800 text-white font-medium py-3 rounded-xl transition-all"
                            >
                                Return to Login
                            </button>
                        </div>
                    ) : (
                        <form onSubmit={handleForgotPasswordSubmit} className="space-y-4">
                            {resetError && (
                                <div className="bg-red-50 border border-red-200 text-red-600 text-xs px-4 py-3 rounded-xl flex items-center gap-2">
                                    <i className="fa-solid fa-circle-exclamation"></i>
                                    <span>{resetError}</span>
                                </div>
                            )}
                            <div className="relative">
                                <i className="fa-regular fa-envelope absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"></i>
                                <input 
                                    type="text" 
                                    value={resetEmail}
                                    onChange={e => setResetEmail(e.target.value)}
                                    placeholder="Email Address"
                                    className="w-full bg-slate-50 border border-slate-200 rounded-xl pl-11 pr-4 py-3 text-slate-800 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-sm"
                                    autoFocus
                                />
                            </div>
                            <button type="submit" className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 rounded-xl shadow-lg shadow-indigo-500/20 transition-all">
                                Send Link
                            </button>
                        </form>
                    )}
                </div>
            ) : (
                <div className="animate-in fade-in slide-in-from-left duration-300 mt-6 pb-4">
                    <div className="text-center mb-8">
                        <h1 className="text-2xl font-bold text-slate-800">Welcome Back</h1>
                        <p className="text-slate-500 text-sm mt-1">Sign in to continue to Jeel AI</p>
                    </div>

                    {error && (
                        <div className="mb-4 bg-red-50 border border-red-200 text-red-600 text-xs px-4 py-3 rounded-xl flex items-center gap-2 animate-in fade-in slide-in-from-top-2">
                            <i className="fa-solid fa-circle-exclamation"></i>
                            <span>{error}</span>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="relative">
                            <i className="fa-regular fa-envelope absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"></i>
                            <input 
                                type="text" 
                                value={email}
                                onChange={e => setEmail(e.target.value)}
                                placeholder="Email Address"
                                className="w-full bg-slate-50 border border-slate-200 rounded-xl pl-11 pr-4 py-3 text-slate-800 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-sm"
                                autoFocus
                            />
                        </div>

                        <div className="relative">
                            <i className="fa-solid fa-lock absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"></i>
                            <input 
                                type="password" 
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                placeholder="Password"
                                className="w-full bg-slate-50 border border-slate-200 rounded-xl pl-11 pr-4 py-3 text-slate-800 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-sm"
                            />
                        </div>
                        
                        <div className="flex justify-end">
                            <button type="button" onClick={() => setIsForgotPassword(true)} className="text-xs text-indigo-600 hover:text-indigo-800 font-medium">
                                Forgot Password?
                            </button>
                        </div>

                        <button type="submit" className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 rounded-xl shadow-lg shadow-indigo-500/20 transition-all transform active:scale-[0.99]">
                            Sign In
                        </button>
                    </form>

                    <div className="mt-8 relative flex items-center justify-center">
                        <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-slate-200"></div></div>
                        <span className="relative bg-white px-4 text-xs text-slate-400 uppercase tracking-wide">Or continue with</span>
                    </div>

                    <div className="mt-6 flex justify-center gap-3">
                         {["google", "facebook", "apple"].map((p) => (
                             <button key={p} onClick={() => handleSocialLogin(p.charAt(0).toUpperCase() + p.slice(1))} className="w-12 h-12 rounded-xl border border-slate-200 flex items-center justify-center text-slate-600 hover:bg-slate-50 hover:border-slate-300 transition-all hover:-translate-y-1">
                                 <i className={`fa-brands fa-${p} text-lg`}></i>
                             </button>
                         ))}
                    </div>
                </div>
            )}
            
            <div className="mt-8 text-center flex justify-center gap-6 text-[11px] text-slate-400 font-medium">
                <a href="https://jeel.ai" target="_blank" className="hover:text-indigo-600 transition-colors">Privacy Policy</a>
                <a href="https://jeel.ai" target="_blank" className="hover:text-indigo-600 transition-colors">Terms of Service</a>
            </div>
        </div>
    </div>
  );
}

// --- Chat View (Unified) ---
const ChatView: React.FC<{ 
  session: ChatSession;
  onUpdateSession: (updatedSession: ChatSession) => void;
  username?: string;
  onAddToLibrary: (item: LibraryItem) => void;
}> = ({ 
  session, 
  onUpdateSession, 
  username,
  onAddToLibrary
}) => {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelType, setModelType] = useState<ChatModelType>(session.mode === "code" ? "pro" : "flash");
  const [selectedVoice, setSelectedVoice] = useState("Kore");
  const [useSearch, setUseSearch] = useState(false);
  const [useMaps, setUseMaps] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [activeMode, setActiveMode] = useState<SpecializedMode>("default");
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [canvasOpen, setCanvasOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const plusMenuRef = useRef<HTMLDivElement>(null);
  const messages = session.messages;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
          if (plusMenuRef.current && !plusMenuRef.current.contains(event.target as Node)) {
              setShowPlusMenu(false);
          }
      };
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSend = async (text: string = input) => {
    if ((!text.trim() && attachedFiles.length === 0) || isLoading) return;

    const userMsg: Message = {
      role: "user",
      content: text,
      type: "text",
      timestamp: Date.now(),
    };
    
    // Optimistic update
    const newMessages = [...messages, userMsg];
    onUpdateSession({ ...session, messages: newMessages, updatedAt: Date.now() });
    
    setInput("");
    setIsLoading(true);
    setShowPlusMenu(false);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const historyContent = newMessages
        .slice(-15) // Only take recent messages
        .map(m => {
           let textPart = m.content;
           if (m.type === 'image') textPart = `[Image generated: ${m.content}]`;
           if (m.type === 'video') textPart = `[Video generated: ${m.content}]`;
           if (m.isToolUse) textPart = `[Tool use: ${m.content}]`;

           return {
              role: m.role,
              parts: [{ text: textPart }]
           };
        });

      const contents = historyContent;
      
      const currentParts: any[] = [];
      if (text.trim()) currentParts.push({ text: text });
      for (const file of attachedFiles) {
        currentParts.push(await fileToGenerativePart(file));
      }
      setAttachedFiles([]); 

      if (contents.length > 0) {
          contents[contents.length - 1] = { role: 'user', parts: currentParts };
      } else {
          contents.push({ role: 'user', parts: currentParts });
      }

      // --- Mode Logic ---
      let selectedModelName = MODEL_MAPPING[modelType];
      let systemInstruction = `You are Jeel AI. User: ${username}. Use markdown.`;
      const tools: any[] = [];

      if (activeMode === "research") {
          selectedModelName = "gemini-3-pro-preview";
          tools.push({ googleSearch: {} });
          systemInstruction += " Perform deep research on the topic. Provide comprehensive, detailed answers with multiple sources.";
      } else if (activeMode === "shopping") {
          selectedModelName = "gemini-2.5-flash"; // fast for shopping
          tools.push({ googleSearch: {} });
          systemInstruction += " You are a shopping assistant. Find the best products, compare prices, and list options with links.";
      } else if (activeMode === "study") {
          systemInstruction += " You are a patient and knowledgeable tutor. Explain concepts simply, use analogies, and help the user learn step-by-step.";
      } else if (activeMode === "image_gen") {
          // If in image gen mode, force the tool use if not implicit
          tools.push({ functionDeclarations: [imageGenTool] });
      }

      // Default toggles override
      if (useSearch && activeMode === "default") tools.push({ googleSearch: {} });

      let toolConfig = undefined;

      // Handle Maps exclusivity and location
      if (useMaps) {
         // Maps can be used with Search, but not other tools (like image/video gen)
         const hasSearch = tools.some(t => t.googleSearch);
         tools.length = 0; // Clear tools to prevent conflicts
         tools.push({ googleMaps: {} });
         if (hasSearch) tools.push({ googleSearch: {} });
         
         // Try to get location
         try {
             const position = await new Promise<GeolocationPosition>((resolve, reject) => {
                 navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 });
             });
             toolConfig = {
                 retrievalConfig: {
                     latLng: {
                         latitude: position.coords.latitude,
                         longitude: position.coords.longitude
                     }
                 }
             };
         } catch (e) {
             console.warn("Location access denied or failed for Maps grounding.");
         }
      } else {
        // If Maps is NOT used, we can add Image/Video gen tools unless explicitly in a mode that avoids them
        if (session.mode !== "code" && activeMode !== "image_gen") {
             tools.push({ functionDeclarations: [imageGenTool, videoGenTool] });
        }
      }

      if (attachedFiles.some(f => f.type.startsWith('video'))) {
        selectedModelName = "gemini-3-pro-preview";
      }

      const config: any = {
        tools: tools.length > 0 ? tools : undefined,
        toolConfig: toolConfig,
        systemInstruction: session.mode === "code" 
          ? "You are an expert software engineer. Follow strict markdown code structure."
          : systemInstruction,
      };

      if (modelType === "thinking") {
        config.thinkingConfig = { thinkingBudget: 32768 };
      }

      const response = await ai.models.generateContent({
        model: selectedModelName,
        contents: contents,
        config,
      });

      const functionCalls = response.functionCalls;
      
      if (functionCalls && functionCalls.length > 0) {
        let currentMessages = [...messages, userMsg]; // Stable reference for sequential updates

        for (const call of functionCalls) {
          if (call.name === "generate_image") {
            const prompt = (call.args as any).prompt;
            
            // 1. Show Loading
            const toolMsg: Message = { role: "model", content: `Generating image for: "${prompt}"...`, type: "text", timestamp: Date.now(), isToolUse: true };
            currentMessages.push(toolMsg);
            onUpdateSession({ ...session, messages: [...currentMessages], updatedAt: Date.now() });
            
            // 2. Execute
            const generatedMsg = await performImageGeneration(prompt);

            // 3. Replace loading with result
            if (generatedMsg) {
                // Find and replace the specific tool message to maintain order/integrity
                const index = currentMessages.indexOf(toolMsg);
                if (index !== -1) {
                    currentMessages[index] = generatedMsg;
                } else {
                    currentMessages.push(generatedMsg);
                }
                onUpdateSession({ ...session, messages: [...currentMessages], updatedAt: Date.now() });
            }

          } else if (call.name === "generate_video") {
            const prompt = (call.args as any).prompt;
            
            // 1. Show Loading
            const toolMsg: Message = { role: "model", content: `Generating video for: "${prompt}"...`, type: "text", timestamp: Date.now(), isToolUse: true };
            currentMessages.push(toolMsg);
            onUpdateSession({ ...session, messages: [...currentMessages], updatedAt: Date.now() });
            
            // 2. Execute
            const generatedMsg = await performVideoGeneration(prompt);

            // 3. Replace loading with result
            if (generatedMsg) {
                const index = currentMessages.indexOf(toolMsg);
                if (index !== -1) {
                    currentMessages[index] = generatedMsg;
                } else {
                    currentMessages.push(generatedMsg);
                }
                onUpdateSession({ ...session, messages: [...currentMessages], updatedAt: Date.now() });
            }
          }
        }
      } else {
        const text = response.text || "I couldn't generate a text response.";
        const groundingMetadata = response.candidates?.[0]?.groundingMetadata;
        
        const modelMsg: Message = {
          role: "model",
          content: text,
          type: "text",
          timestamp: Date.now(),
          groundingMetadata
        };
        
        onUpdateSession({ ...session, messages: [...messages, userMsg, modelMsg], updatedAt: Date.now() });
      }

    } catch (e: any) {
      console.error(e);
      const errorMsg: Message = {
        role: "model",
        content: `Error: ${e.message || "Something went wrong."}`,
        type: "text",
        timestamp: Date.now()
      };
      onUpdateSession({ ...session, messages: [...messages, userMsg, errorMsg], updatedAt: Date.now() });
    } finally {
      setIsLoading(false);
      if (activeMode === "image_gen") setActiveMode("default"); // reset after use
    }
  };

  const performImageGeneration = async (prompt: string): Promise<Message | null> => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-image",
        contents: { parts: [{ text: prompt }] },
        config: { imageConfig: { aspectRatio: "1:1" } }
      });
      
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            const url = `data:image/png;base64,${part.inlineData.data}`;
            
            // Add to library
            onAddToLibrary({
                id: Date.now().toString(),
                type: 'image',
                url: url,
                prompt: prompt,
                timestamp: Date.now()
            });

            return { role: "model", content: prompt, type: "image", imageData: url, timestamp: Date.now() };
          }
        }
      }
      return { role: "model", content: "Sorry, I couldn't generate the image.", type: "text", timestamp: Date.now() };
    } catch (e) {
       return { role: "model", content: "Error generating image.", type: "text", timestamp: Date.now() };
    }
  };

  const performVideoGeneration = async (prompt: string): Promise<Message | null> => {
    try {
      await ensureApiKey();
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      let operation = await ai.models.generateVideos({
        model: 'veo-3.1-fast-generate-preview',
        prompt: prompt,
        config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
      });

      let attempts = 0;
      // Increased timeout to ~4 minutes (60 * 4s) for video generation
      while (!operation.done && attempts < 60) {
        await new Promise(r => setTimeout(r, 4000));
        operation = await ai.operations.getVideosOperation({ operation });
        attempts++;
      }

      const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
      if (videoUri) {
        const res = await fetch(`${videoUri}&key=${process.env.API_KEY}`);
        const blob = await res.blob();
        
        // Convert to base64 for persistence
        const base64Video = await blobToBase64(blob);

        // Add to library
        onAddToLibrary({
            id: Date.now().toString(),
            type: 'video',
            url: base64Video,
            prompt: prompt,
            timestamp: Date.now()
        });
        
        return { role: "model", content: prompt, type: "video", videoUrl: base64Video, timestamp: Date.now() };
      } else {
        throw new Error("Timeout or no video");
      }
    } catch (e) {
      console.error(e);
      return { role: "model", content: "Error generating video.", type: "text", timestamp: Date.now() };
    }
  };

  const getLastModelMessage = () => {
      const modelMsgs = messages.filter(m => m.role === "model" && m.type === "text");
      return modelMsgs.length > 0 ? modelMsgs[modelMsgs.length - 1] : null;
  };

  const lastMsg = getLastModelMessage();

  return (
    <div className="flex h-full bg-[#0f1117] overflow-hidden">
      {/* Messages Area */}
      <div className={`flex flex-col h-full flex-1 min-w-0 transition-all duration-300 ${canvasOpen ? 'mr-0' : ''}`}>
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 min-h-0">
            {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 animate-in fade-in zoom-in-95 duration-500">
                <div className="bg-slate-800/50 p-6 rounded-3xl mb-8 shadow-2xl shadow-black/20">
                    <AppLogo className="w-20 h-20 grayscale opacity-80" />
                </div>
                <h2 className="text-2xl font-bold text-slate-200 mb-2">How can I help you today?</h2>
                <p className="text-slate-500 mb-8 text-center max-w-md">I'm Jeel AI. I can generate code, images, videos, and search the web for you.</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl px-4">
                    {SUGGESTIONS.map((s, i) => (
                        <button 
                            key={i} 
                            onClick={() => handleSend(s.prompt)}
                            className="flex items-center gap-4 p-4 rounded-2xl bg-slate-800/40 hover:bg-slate-800 border border-slate-700/50 hover:border-indigo-500/50 transition-all group text-left"
                        >
                            <div className="w-10 h-10 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-300 group-hover:bg-indigo-500/20 group-hover:text-indigo-400 transition-colors">
                                <i className={`${s.icon}`}></i>
                            </div>
                            <div>
                                <div className="font-medium text-slate-300 group-hover:text-white text-sm">{s.label}</div>
                                <div className="text-xs text-slate-500 group-hover:text-slate-400 truncate max-w-[180px]">{s.prompt}</div>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
            )}
            {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} selectedVoice={selectedVoice} />
            ))}
            {isLoading && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 md:p-6 bg-[#0f1117] border-t border-slate-800/50 relative z-10">
            <div className="max-w-4xl mx-auto relative">
                
                {/* Active Mode Indicator */}
                {activeMode !== "default" && (
                    <div className="absolute -top-10 left-0 bg-indigo-600/20 border border-indigo-500/30 text-indigo-300 px-3 py-1 rounded-full text-xs font-medium flex items-center gap-2 animate-in fade-in slide-in-from-bottom-1">
                        <i className={`fa-solid ${activeMode === 'research' ? 'fa-magnifying-glass-chart' : activeMode === 'shopping' ? 'fa-bag-shopping' : activeMode === 'study' ? 'fa-graduation-cap' : 'fa-wand-magic-sparkles'}`}></i>
                        <span className="capitalize">{activeMode.replace('_', ' ')} Mode Active</span>
                        <button onClick={() => setActiveMode("default")} className="hover:text-white ml-1"><i className="fa-solid fa-xmark"></i></button>
                    </div>
                )}

                <div className="flex gap-2 bg-slate-800/40 p-1.5 rounded-[1.5rem] border border-slate-700/50 focus-within:border-indigo-500/50 focus-within:bg-slate-800/80 focus-within:ring-1 focus-within:ring-indigo-500/30 transition-all shadow-lg shadow-black/10 items-end">
                
                {/* Plus Menu Button */}
                <div className="relative" ref={plusMenuRef}>
                    <button 
                        onClick={() => setShowPlusMenu(!showPlusMenu)} 
                        className={`w-9 h-9 flex items-center justify-center transition-all rounded-full shrink-0 mb-0.5 ${showPlusMenu ? 'bg-slate-700 text-white rotate-45' : 'text-slate-400 hover:text-indigo-400 hover:bg-indigo-500/10'}`}
                        title="Add..."
                    >
                        <i className="fa-solid fa-plus text-lg"></i>
                    </button>

                    {/* Power Menu Popover */}
                    {showPlusMenu && (
                        <div className="absolute bottom-14 left-0 w-64 bg-[#1e293b] border border-slate-700 rounded-2xl shadow-2xl p-2 z-50 animate-in fade-in zoom-in-95 duration-200 origin-bottom-left">
                            <div className="grid grid-cols-2 gap-2">
                                <button onClick={() => fileInputRef.current?.click()} className="flex flex-col items-center gap-1.5 p-3 rounded-xl hover:bg-slate-700/50 text-slate-300 hover:text-white transition-colors">
                                    <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400"><i className="fa-regular fa-file"></i></div>
                                    <span className="text-[10px] font-medium">Upload File</span>
                                </button>
                                <button onClick={() => fileInputRef.current?.click()} className="flex flex-col items-center gap-1.5 p-3 rounded-xl hover:bg-slate-700/50 text-slate-300 hover:text-white transition-colors">
                                    <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400"><i className="fa-regular fa-image"></i></div>
                                    <span className="text-[10px] font-medium">Add Photo</span>
                                </button>
                                <button onClick={() => { setActiveMode("image_gen"); setShowPlusMenu(false); }} className="flex flex-col items-center gap-1.5 p-3 rounded-xl hover:bg-slate-700/50 text-slate-300 hover:text-white transition-colors">
                                    <div className="w-8 h-8 rounded-full bg-pink-500/20 flex items-center justify-center text-pink-400"><i className="fa-solid fa-wand-magic-sparkles"></i></div>
                                    <span className="text-[10px] font-medium">Create Image</span>
                                </button>
                                <button onClick={() => { setCanvasOpen(!canvasOpen); setShowPlusMenu(false); }} className="flex flex-col items-center gap-1.5 p-3 rounded-xl hover:bg-slate-700/50 text-slate-300 hover:text-white transition-colors">
                                    <div className="w-8 h-8 rounded-full bg-orange-500/20 flex items-center justify-center text-orange-400"><i className="fa-solid fa-table-columns"></i></div>
                                    <span className="text-[10px] font-medium">Canvas</span>
                                </button>
                            </div>
                            
                            <div className="h-px bg-slate-700/50 my-2 mx-1"></div>
                            <div className="px-2 py-1 text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-1">Modes</div>
                            
                            <button onClick={() => { setActiveMode("research"); setShowPlusMenu(false); }} className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-slate-700/50 text-left transition-colors">
                                <i className="fa-solid fa-magnifying-glass-chart text-teal-400 w-4"></i>
                                <div>
                                    <div className="text-xs font-medium text-slate-200">Deep Research</div>
                                    <div className="text-[10px] text-slate-500">Pro model + Web Search</div>
                                </div>
                            </button>
                            <button onClick={() => { setActiveMode("shopping"); setShowPlusMenu(false); }} className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-slate-700/50 text-left transition-colors">
                                <i className="fa-solid fa-bag-shopping text-rose-400 w-4"></i>
                                <div>
                                    <div className="text-xs font-medium text-slate-200">Shopping</div>
                                    <div className="text-[10px] text-slate-500">Find products & prices</div>
                                </div>
                            </button>
                            <button onClick={() => { setActiveMode("study"); setShowPlusMenu(false); }} className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-slate-700/50 text-left transition-colors">
                                <i className="fa-solid fa-graduation-cap text-yellow-400 w-4"></i>
                                <div>
                                    <div className="text-xs font-medium text-slate-200">Study & Learn</div>
                                    <div className="text-[10px] text-slate-500">Tutor & Explanation</div>
                                </div>
                            </button>
                             <button onClick={() => { setModelType("thinking"); setShowPlusMenu(false); }} className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-slate-700/50 text-left transition-colors">
                                <i className="fa-solid fa-brain text-violet-400 w-4"></i>
                                <div>
                                    <div className="text-xs font-medium text-slate-200">Thinking</div>
                                    <div className="text-[10px] text-slate-500">Complex reasoning</div>
                                </div>
                            </button>
                        </div>
                    )}
                </div>

                <input 
                    type="file" 
                    multiple 
                    ref={fileInputRef} 
                    className="hidden" 
                    onChange={(e) => {
                        if (e.target.files) {
                            setAttachedFiles(prev => [...prev, ...Array.from(e.target.files!)]);
                            e.target.value = ''; // Reset input so same file can be selected again
                        }
                    }} 
                />
                
                <div className="flex-1 flex flex-col justify-end">
                    {attachedFiles.length > 0 && (
                        <div className="flex gap-2 mb-2 overflow-x-auto pb-1">
                            {attachedFiles.map((f, i) => (
                            <div key={i} className="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg text-xs flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2">
                                <i className={`fa-solid ${f.type.startsWith('image') ? 'fa-image text-purple-400' : 'fa-file text-blue-400'}`}></i>
                                <span className="text-slate-300 truncate max-w-[100px]">{f.name}</span>
                                <button onClick={() => setAttachedFiles(prev => prev.filter((_, idx) => idx !== i))} className="hover:text-red-400"><i className="fa-solid fa-xmark"></i></button>
                            </div>
                            ))}
                        </div>
                    )}
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                             if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                             }
                        }}
                        placeholder={activeMode !== 'default' ? `Message Jeel (${activeMode} mode)...` : "Message Jeel AI..."}
                        className="w-full bg-transparent text-slate-200 placeholder-slate-500 focus:outline-none text-[15px] resize-none py-2 max-h-[120px]"
                        rows={1}
                        style={{ height: 'auto', minHeight: '24px' }} 
                        onInput={(e) => {
                            (e.target as HTMLTextAreaElement).style.height = 'auto';
                            (e.target as HTMLTextAreaElement).style.height = (e.target as HTMLTextAreaElement).scrollHeight + 'px';
                        }}
                    />
                </div>

                <div className="flex flex-col justify-end gap-2 pb-0.5">
                    {/* Inline Quick Toggles */}
                    <div className="flex gap-1">
                         <button 
                            onClick={() => {
                                if (useMaps) setUseMaps(false); // Exclusive toggle
                                setUseSearch(!useSearch);
                            }}
                            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${useSearch ? 'text-blue-400 bg-blue-500/10' : 'text-slate-500 hover:text-slate-300'}`}
                            title="Web Search"
                         >
                             <i className="fa-brands fa-google"></i>
                         </button>
                         <button 
                            onClick={() => {
                                if (useSearch) setUseSearch(false); // Exclusive toggle logic preference
                                setUseMaps(!useMaps);
                            }}
                            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${useMaps ? 'text-green-400 bg-green-500/10' : 'text-slate-500 hover:text-slate-300'}`}
                            title="Google Maps"
                         >
                             <i className="fa-solid fa-map-location-dot"></i>
                         </button>
                    </div>
                    <button 
                        onClick={() => handleSend()}
                        disabled={isLoading || (!input.trim() && attachedFiles.length === 0)}
                        className={`w-9 h-9 rounded-full flex items-center justify-center transition-all shrink-0 ${
                            isLoading || (!input.trim() && attachedFiles.length === 0) 
                            ? "bg-slate-700 text-slate-500 cursor-not-allowed" 
                            : "bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/30 hover:scale-105 active:scale-95"
                        }`}
                    >
                        <i className={`fa-solid ${isLoading ? 'fa-circle-notch fa-spin' : 'fa-arrow-up'}`}></i>
                    </button>
                </div>
                </div>
                <p className="text-center text-slate-600 text-[10px] mt-3">Jeel AI can make mistakes. Check important info.</p>
            </div>
          </div>
      </div>
      
      {/* Canvas / Side Panel */}
      {canvasOpen && (
        <div className="w-[45%] border-l border-slate-800 bg-[#0b0c11] flex flex-col animate-in slide-in-from-right duration-300 relative z-20">
            <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-[#0f1117]">
                <div className="flex items-center gap-2 text-slate-200 font-semibold">
                    <i className="fa-solid fa-table-columns text-indigo-500"></i>
                    <span>Canvas</span>
                </div>
                <button onClick={() => setCanvasOpen(false)} className="text-slate-500 hover:text-white transition-colors">
                    <i className="fa-solid fa-xmark"></i>
                </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 bg-[#0f1117] custom-scrollbar">
                {lastMsg ? (
                    <div className="prose prose-invert max-w-none">
                        <FormattedContent message={lastMsg} />
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500">
                        <i className="fa-regular fa-file-code text-4xl mb-3 opacity-30"></i>
                        <p className="text-sm">Content generated by Jeel will appear here.</p>
                    </div>
                )}
            </div>
        </div>
      )}
    </div>
  );
};

const App = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [user, setUser] = useState<string | null>(localStorage.getItem("jeel_user"));
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const profileRef = useRef<HTMLDivElement>(null);
  
  // App State
  const [view, setView] = useState<'chat' | 'library'>('chat');
  
  // Library State Management
  const [library, setLibrary] = useState<LibraryItem[]>(() => {
    try {
        const saved = localStorage.getItem("jeel_library");
        return saved ? JSON.parse(saved) : [];
    } catch (e) {
        console.error("Failed to load library", e);
        return [];
    }
  });

  // Session State Management
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
        const saved = localStorage.getItem("jeel_sessions");
        return saved ? JSON.parse(saved) : [];
    } catch (e) {
        console.error("Failed to load sessions", e);
        return [];
    }
  });
  
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  useEffect(() => {
    try {
        localStorage.setItem("jeel_sessions", JSON.stringify(sessions));
    } catch (e) {
        console.error("Failed to save sessions", e);
    }
  }, [sessions]);

  useEffect(() => {
    try {
        localStorage.setItem("jeel_library", JSON.stringify(library));
    } catch (e) {
        console.error("Failed to save library - quota exceeded?", e);
    }
  }, [library]);

  const addToLibrary = (item: LibraryItem) => {
    setLibrary(prev => [item, ...prev]);
  };

  const deleteFromLibrary = (id: string) => {
    setLibrary(prev => prev.filter(item => item.id !== id));
  };

  // Click outside listener for profile menu
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setIsProfileOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Create default session if none exists
  useEffect(() => {
    if (sessions.length === 0) {
        handleNewChat('chat');
    } else if (!activeSessionId) {
        // Restore last session or first one
        setActiveSessionId(sessions[0].id);
    }
  }, []); // Only on mount

  useEffect(() => {
    let lastWidth = window.innerWidth;
    const handleResize = () => {
      // Only change sidebar state if we cross the 768px threshold
      const newWidth = window.innerWidth;
      if (newWidth !== lastWidth) {
          if (newWidth < 768 && lastWidth >= 768) setIsSidebarOpen(false);
          if (newWidth >= 768 && lastWidth < 768) setIsSidebarOpen(true);
          lastWidth = newWidth;
      }
    };
    
    // Initial check: if on mobile, default to closed
    if (window.innerWidth < 768) setIsSidebarOpen(false);

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleLogin = (name: string) => {
    localStorage.setItem("jeel_user", name);
    setUser(name);
  };

  const handleLogout = () => {
    localStorage.removeItem("jeel_user");
    setUser(null);
  };
  
  const handleNewChat = (mode: 'chat' | 'code' = 'chat') => {
    const newSession: ChatSession = {
        id: Date.now().toString(),
        title: "New Chat",
        messages: [],
        mode: mode,
        updatedAt: Date.now()
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setSearchTerm("");
    setView('chat'); // Switch to chat view
    
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const updateSession = (updated: ChatSession) => {
      setSessions(prev => {
          const newSessions = prev.map(s => s.id === updated.id ? updated : s);
          
          // Auto-title logic: If title is "New Chat" and we have a user message, update title
          if (updated.title === "New Chat" && updated.messages.length > 0) {
              const firstUserMsg = updated.messages.find(m => m.role === "user");
              if (firstUserMsg) {
                  // truncate to ~30 chars
                  const text = firstUserMsg.content;
                  const newTitle = text.length > 30 ? text.substring(0, 30) + "..." : text;
                  updated.title = newTitle;
              }
          }
          return newSessions;
      });
  };

  const deleteSession = (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      setSessions(prev => {
          const filtered = prev.filter(s => s.id !== id);
          if (filtered.length === 0) {
              // Should create a new one immediately, but state update is async.
              // Logic handles in useEffect above or we do it here.
              // Let's do it here for instant feedback.
              const newS: ChatSession = { id: Date.now().toString(), title: "New Chat", messages: [], mode: 'chat', updatedAt: Date.now() };
              setActiveSessionId(newS.id);
              setView('chat');
              return [newS];
          }
          if (activeSessionId === id) {
              setActiveSessionId(filtered[0].id);
          }
          return filtered;
      });
  };

  if (!user) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
  const filteredSessions = sessions.filter(s => s.title.toLowerCase().includes(searchTerm.toLowerCase()));

  return (
    <div className="flex h-[100dvh] bg-[#0f1117] text-slate-200 overflow-hidden font-sans selection:bg-indigo-500/30">
      {/* Sidebar */}
      <div className={`${isSidebarOpen ? "w-[280px] translate-x-0" : "w-0 -translate-x-full opacity-0"} transition-all duration-300 bg-[#0b0c11] border-r border-slate-800 flex flex-col absolute md:relative z-30 h-full shadow-2xl overflow-hidden top-0 left-0`}>
        <div className="p-6 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <AppLogo />
            <h1 className="font-bold text-xl tracking-tight text-white">Jeel AI</h1>
          </div>
          {/* Mobile Close Button */}
          <button 
             onClick={() => setIsSidebarOpen(false)} 
             className="md:hidden text-slate-400 hover:text-white p-1 rounded-lg"
          >
             <i className="fa-solid fa-xmark text-lg"></i>
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto py-2 px-3">
            <button 
                onClick={() => handleNewChat('chat')}
                className="w-full flex items-center gap-3 px-4 py-3 mb-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 transition-all font-medium text-sm hover:scale-[1.02] active:scale-[0.98]"
            >
                <i className="fa-solid fa-plus w-5 text-center"></i>
                <span>New Chat</span>
            </button>
            
            <div className="px-1 mb-4">
               <div className="relative group">
                   <i className="fa-solid fa-magnifying-glass absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs group-focus-within:text-indigo-400 transition-colors"></i>
                   <input 
                      type="text" 
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      placeholder="Search chats..." 
                      className="w-full bg-slate-900/50 border border-slate-700/50 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-indigo-500/50 focus:bg-slate-900 focus:ring-1 focus:ring-indigo-500/20 transition-all placeholder:text-slate-600"
                   />
                   {searchTerm && (
                       <button onClick={() => setSearchTerm("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-400">
                           <i className="fa-solid fa-xmark text-xs"></i>
                       </button>
                   )}
               </div>
            </div>

            <button 
                onClick={() => { setView('library'); if(window.innerWidth < 768) setIsSidebarOpen(false); }}
                className={`w-full flex items-center gap-3 px-4 py-3 mb-6 rounded-xl transition-all font-medium text-sm group ${view === 'library' ? 'bg-slate-800 text-white border border-slate-700' : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'}`}
            >
                <i className={`fa-solid fa-book-open w-5 text-center ${view === 'library' ? 'text-indigo-400' : 'text-slate-500 group-hover:text-white'}`}></i>
                <span>Library</span>
            </button>
            
          <div className="px-4 mb-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest flex justify-between items-center">
             <span>History</span>
          </div>
          <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-280px)] mb-6 custom-scrollbar pr-1">
             {filteredSessions.length > 0 ? (
                 filteredSessions.map(s => (
                    <div 
                       key={s.id} 
                       onClick={() => { setActiveSessionId(s.id); setView('chat'); if(window.innerWidth < 768) setIsSidebarOpen(false); }}
                       className={`group relative flex items-center gap-3 px-4 py-2.5 rounded-xl cursor-pointer transition-all ${activeSessionId === s.id && view === 'chat' ? "bg-slate-800/80 text-white border border-slate-700" : "text-slate-400 hover:bg-slate-800/40 hover:text-slate-200"}`}
                    >
                       <i className={`text-xs w-5 text-center ${s.mode === 'code' ? "fa-solid fa-code text-purple-400" : "fa-regular fa-message text-indigo-400"}`}></i>
                       <span className="text-xs font-medium truncate flex-1">{s.title}</span>
                       
                       <button 
                          onClick={(e) => deleteSession(e, s.id)}
                          className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-all"
                       >
                          <i className="fa-solid fa-trash text-xs"></i>
                       </button>
                    </div>
                 ))
             ) : (
                <div className="text-center py-8 text-slate-600 text-xs italic">
                    {searchTerm ? "No chats found" : "No history yet"}
                </div>
             )}
          </div>

        </div>
        
        <div className="p-4 border-t border-slate-800 bg-[#0b0c11] relative" ref={profileRef}>
           {/* Profile Menu Popover */}
           {isProfileOpen && (
             <div className="absolute bottom-full left-4 right-4 mb-2 bg-[#1e293b] border border-slate-700 rounded-xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-2 z-50">
                <div className="p-3 border-b border-slate-700/50 bg-slate-800/50">
                    <p className="font-bold text-white text-sm">{user}</p>
                    <p className="text-[10px] text-slate-400">Free Plan</p>
                </div>
                
                <div className="p-1">
                   <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-amber-400 hover:bg-slate-700/50 rounded-lg transition-colors text-left">
                       <i className="fa-solid fa-crown w-4 text-center"></i> Upgrade Plan
                   </button>
                   <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors text-left">
                       <i className="fa-solid fa-gear w-4 text-center"></i> Settings
                   </button>
                   <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors text-left">
                       <i className="fa-solid fa-circle-question w-4 text-center"></i> Help & FAQ
                   </button>
                   <div className="h-px bg-slate-700/50 my-1"></div>
                   <button 
                     onClick={handleLogout}
                     className="w-full flex items-center gap-3 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors text-left"
                   >
                       <i className="fa-solid fa-arrow-right-from-bracket w-4 text-center"></i> Log out
                   </button>
                </div>
             </div>
           )}

           {/* Trigger Button */}
           <button 
             onClick={() => setIsProfileOpen(!isProfileOpen)}
             className={`w-full flex items-center gap-3 p-2 rounded-xl border transition-all ${isProfileOpen ? 'bg-slate-800 border-indigo-500/30' : 'bg-slate-800/40 border-slate-700/30 hover:bg-slate-800 hover:border-slate-600'}`}
           >
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold text-xs shadow-md">
                 {user.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 text-left overflow-hidden">
                 <p className="text-xs font-bold text-white truncate">{user}</p>
                 <p className="text-[10px] text-slate-400 truncate">Free Account</p>
              </div>
              <i className="fa-solid fa-ellipsis-vertical text-slate-500 text-xs"></i>
           </button>
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col relative w-full h-full bg-[#0f1117]">
        <header className="h-16 border-b border-slate-800/50 flex items-center px-4 justify-between bg-[#0f1117]/80 backdrop-blur-md z-20">
           <div className="flex items-center gap-3">
             <button 
               onClick={() => setIsSidebarOpen(!isSidebarOpen)}
               className="p-2.5 -ml-2 text-slate-400 hover:text-white rounded-xl hover:bg-slate-800 transition-colors"
             >
               <i className="fa-solid fa-bars"></i>
             </button>
             <span className="font-semibold text-slate-200 text-sm">
               {view === 'library' ? 'My Library' : (activeSession ? activeSession.title : "Jeel AI")}
             </span>
             <span className="px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 text-[10px] font-medium border border-indigo-500/20">Beta</span>
           </div>
           
           <div className="flex items-center gap-4">
              <a href="https://github.com/google/genai" target="_blank" className="text-slate-400 hover:text-white transition-colors" title="View Source">
                <i className="fa-brands fa-github text-lg"></i>
              </a>
           </div>
        </header>

        <main className="flex-1 overflow-hidden relative flex flex-col min-h-0">
          {view === 'library' ? (
             <LibraryView items={library} onDelete={deleteFromLibrary} />
          ) : activeSession ? (
             <ChatView 
                key={activeSession.id} 
                session={activeSession} 
                onUpdateSession={updateSession}
                username={user}
                onAddToLibrary={addToLibrary}
             />
          ) : null}
        </main>
      </div>
      
      {/* Overlay for mobile sidebar */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-20 md:hidden backdrop-blur-sm transition-opacity"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);