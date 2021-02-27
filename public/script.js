const socket = io('/')
const videoGrid = document.getElementById('video-grid')
const connectionPeer=new Peer(undefined,{
    host: '/',
    port: '3001'
})
const myVideo= document.createElement('video')
myVideo.muted = true
const connections={}
navigator.mediaDevices.getUserMedia({
    video: true,
    audio: true
}).then(stream => {
  addVidStream(myVideo, stream)
  connectionPeer.on('call', call=>{
      call.answer(stream)
      const video =document.createElement('video')
      call.on('stream', userVideoStream=> {
          addVidStream(video, userVideoStream)
      })
  })
  socket.on('user-connected', userId=> {
    connectToNewUser(userId, stream)
})
})
socket.on('user-disconnected', userId=>{
    if(connections[userId]) connections[userId].close()
})

connectionPeer.on('open', id=> {
    socket.emit('join-room', ROOM_ID, id)
})

function connectToNewUser(userId, stream){
    const call = connectionPeer.call(userId, stream)
    const video = document.createElement('video')
    call.on('stream', userVideoStream=>{
        addVidStream(video, userVideoStream)
    })
    call.on('close', ()=> {
        video.remove()
    })
    connections[userId]=call
}

function addVidStream(video, stream){
    video.srcObject = stream
    video.addEventListener('loadedmetadata', ()=> {
        video.play()
    })
    videoGrid.append(video)
}