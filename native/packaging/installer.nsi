; The Windows installer.
;
; Driven by tools/make_installer.sh, which passes the version and the
; paths in rather than letting this file guess them:
;
;   makensis -DVERSION=x.y.z -DSRCDIR=<staged tree> -DLICENSEFILE=<file> \
;            -DOUTFILE=<setup.exe> installer.nsi
;
; NSIS rather than WiX for one reason that matters: WiX describes an MSI,
; and an MSI's component/GUID model wants every file listed with a stable
; identity. What is being installed here is "whatever windeployqt decided
; the app needs", which changes with the Qt version -- so the accurate
; description is a directory, and that is what File /r is.
;
; Per-machine, under Program Files, which is why it asks for
; administrator. A per-user install would avoid the elevation prompt, but
; it also puts the application somewhere other software (and the user)
; will not look, and it cannot be shared between accounts on a shack PC.

Unicode true
!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"   ; GetSize, for the Add/Remove Programs entry

!define UNINST_KEY \
  "Software\Microsoft\Windows\CurrentVersion\Uninstall\SSTVAE"

!ifndef VERSION
  !error "VERSION is required (see tools/make_installer.sh)"
!endif
!ifndef SRCDIR
  !error "SRCDIR is required (see tools/make_installer.sh)"
!endif
!ifndef OUTFILE
  !error "OUTFILE is required (see tools/make_installer.sh)"
!endif

Name "SSTVAE ${VERSION}"
OutFile "${OUTFILE}"
InstallDir "$PROGRAMFILES64\SSTVAE"
; Reuse the previous location on an upgrade, so a user who chose another
; drive is not silently given a second copy under Program Files.
InstallDirRegKey HKLM "Software\SSTVAE" "InstallDir"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "SSTVAE"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}.0"
VIAddVersionKey "FileDescription" "SSTVAE installer"
VIAddVersionKey "LegalCopyright" "Artistic License 2.0"

!define MUI_ICON "${__FILEDIR__}\sstvae.ico"
!define MUI_UNICON "${__FILEDIR__}\sstvae.ico"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!ifdef LICENSEFILE
  !insertmacro MUI_PAGE_LICENSE "${LICENSEFILE}"
!endif
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
; Offer to launch it: the first thing an operator wants to do after
; installing is see whether it starts.
!define MUI_FINISHPAGE_RUN "$INSTDIR\sstvae-gui.exe"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "SSTVAE (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; The whole staged tree: the executable, our two pinned libraries, and
  ; everything windeployqt put beside them. Windows searches the .exe's
  ; own directory first, so this flat layout needs no PATH entry.
  ; `\*` and not `\*.*`: the latter is a DOS-ism that skips any file with
  ; no extension, and one of those going missing from a Qt deployment is
  ; not visible until the app cannot start.
  File /r "${SRCDIR}\*"

  WriteRegStr HKLM "Software\SSTVAE" "InstallDir" "$INSTDIR"

  CreateDirectory "$SMPROGRAMS\SSTVAE"
  CreateShortcut "$SMPROGRAMS\SSTVAE\SSTVAE.lnk" "$INSTDIR\sstvae-gui.exe"
  CreateShortcut "$SMPROGRAMS\SSTVAE\Uninstall SSTVAE.lnk" "$INSTDIR\uninstall.exe"

  ; What makes it appear in Settings > Apps, with a size and a publisher,
  ; and what gives Windows a documented way to remove it. Without this
  ; the only uninstaller is the one in the install directory, which is
  ; the first place a user deletes by hand.
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "SSTVAE"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\sstvae-gui.exe"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "SSTVAE"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "${UNINST_KEY}" "QuietUninstallString" \
    '"$INSTDIR\uninstall.exe" /S'
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${UNINST_KEY}" "EstimatedSize" "$0"

  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section /o "Desktop shortcut" SecDesktop
  CreateShortcut "$DESKTOP\SSTVAE.lnk" "$INSTDIR\sstvae-gui.exe"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} \
    "The application, Qt, Hamlib and the ONNX runtime."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} \
    "Put a shortcut on the desktop as well as in the Start Menu."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function .onInit
  ; 64-bit only, matching what CI builds. Saying so here beats a
  ; successful install that dies with a "not a valid Win32 application"
  ; dialog, which reads as a corrupt download.
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "SSTVAE needs 64-bit Windows."
    Abort
  ${EndIf}
  SetRegView 64
FunctionEnd

Section "Uninstall"
  SetRegView 64
  ; The model cache is deliberately left behind: it lives in the user's
  ; own cache directory, it is shared with anything else that fetched the
  ; same artifacts, and re-downloading it after a reinstall is 9-21 MB the
  ; operator did not ask to spend.
  Delete "$DESKTOP\SSTVAE.lnk"
  Delete "$SMPROGRAMS\SSTVAE\SSTVAE.lnk"
  Delete "$SMPROGRAMS\SSTVAE\Uninstall SSTVAE.lnk"
  RMDir "$SMPROGRAMS\SSTVAE"
  ; RMDir /r on $INSTDIR, which is only safe because the directory is
  ; ours: it is written by this installer and nothing else installs into
  ; it. Guarded anyway -- an empty or unset $INSTDIR would make that line
  ; delete the drive root.
  ${If} $INSTDIR != ""
    RMDir /r "$INSTDIR"
  ${EndIf}
  DeleteRegKey HKLM "${UNINST_KEY}"
  DeleteRegKey HKLM "Software\SSTVAE"
SectionEnd
