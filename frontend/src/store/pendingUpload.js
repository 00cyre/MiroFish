/**
 * Temporary storage for files and requirements pending upload.
 * Used when the user clicks "Start Engine" on the homepage and is immediately
 * redirected to the Process page, where the actual API call is made.
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  simulationRequirement: '',
  promptText: '',
  isPending: false
})

export function setPendingUpload(files, requirement, promptText = '') {
  state.files = files
  state.simulationRequirement = requirement
  state.promptText = promptText
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    simulationRequirement: state.simulationRequirement,
    promptText: state.promptText,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.simulationRequirement = ''
  state.promptText = ''
  state.isPending = false
}

export default state
