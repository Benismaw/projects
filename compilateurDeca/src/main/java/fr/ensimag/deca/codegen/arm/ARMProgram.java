package fr.ensimag.deca.codegen.arm;

import fr.ensimag.ARM.ARMInstruction;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.Instructions.ARMDiv;
import fr.ensimag.ARM.Instructions.ARMPrintFloat;
import fr.ensimag.ARM.Instructions.ARMPrintInt;
import fr.ensimag.ARM.Instructions.ARMPrintString;
import fr.ensimag.ARM.Instructions.ARMPrintln;
import fr.ensimag.ARM.Instructions.ARMReadFloat;
import fr.ensimag.ARM.Instructions.ARMReadInt;
import fr.ensimag.ARM.ARMComment;

public class ARMProgram {
    private final List<String> dataSection = new ArrayList<>();
    private final List<String> rawText = new ArrayList<>();
    private final List<String> textPreamble = new ArrayList<>();
    private final List<ARMInstruction> instructions = new ArrayList<>();
    private boolean printIntAdded = false;
    private boolean printFloatAdded = false;
    private boolean printStringAdded = false;
    private boolean divAdded = false;
    private boolean printLnAdded = false;
    private boolean readIntAdded = false;
    private boolean readFloatAdded = false;


    public ARMProgram() {
        textPreamble.add(".text");
        textPreamble.add(".global main");
        textPreamble.add("main:");
    }

    public void addInstruction(ARMInstruction instruction) {
        if (instruction instanceof ARMPrintInt) {
            ensurePrintInt();
        }

        if (instruction instanceof ARMPrintFloat) {
            ensurePrintFloat();
        }

        if (instruction instanceof ARMPrintString) {
            ensurePrintString();
        }

        if (instruction instanceof ARMDiv) {
            ensureDiv();
        }

        if (instruction instanceof ARMPrintln) {
            ensurePrintLn();
        }

        if (instruction instanceof ARMReadInt) {
            ensureReadInt();
        }

        if (instruction instanceof ARMReadFloat) {
            ensureReadFloat();
        }
        
        instructions.add(instruction);
    }

    public void addData(String line) {
        dataSection.add(line);
    }

    public void addLabel(String label) {
        instructions.add(new ARMLabel(label));
    }

    public void addComment(String comment) {
        instructions.add(new ARMComment(comment));
    }

    public void addRaw(String asm) {
        for (String line : asm.split("\n")) {
            rawText.add(line);
        }
    }

    public void ensurePrintInt() {
        if (printIntAdded) return;
        printIntAdded = true;

        // section data du runtime print_int
    addData("__print_buf: .space 12");
    addData("__pow10_table:");
    addData("    .word 1000000000");
    addData("    .word 100000000");
    addData("    .word 10000000");
    addData("    .word 1000000");
    addData("    .word 100000");
    addData("    .word 10000");
    addData("    .word 1000");
    addData("    .word 100");
    addData("    .word 10");
    addData("    .word 0");

    addData("__min_int_str: .ascii \"-2147483648\"");

    // code du runtime print
    addRaw("""
__print_int:
    push {r1-r7, lr}

    @ Vérifier si R0 == -2147483648 (0x80000000)
    mov r2, #1
    lsl r2, r2, #31     @ R2 devient 0x80000000 (le bit de signe seul)
    cmp r0, r2
    beq __pi_is_min_int @ Si égal, on saute au cas spécial

    ldr r1, =__print_buf
    mov r3, #0
    mov r4, #12

__pi_clear_buf:
    strb r3, [r1], #1
    subs r4, r4, #1
    bne __pi_clear_buf

    ldr r1, =__print_buf
    mov r2, r0
    mov r6, #0

    cmp r2, #0
    bge __pi_loop_start
    mov r3, #45
    strb r3, [r1], #1
    rsb r2, r2, #0

__pi_loop_start:
    ldr r7, =__pow10_table

__pi_digit_loop:
    ldr r3, [r7], #4
    cmp r3, #0
    beq __pi_last_digit

    mov r4, #0

__pi_sub_loop:
    cmp r2, r3
    blt __pi_emit_digit
    sub r2, r2, r3
    add r4, r4, #1
    b __pi_sub_loop

__pi_emit_digit:
    cmp r6, #0
    bne __pi_write_digit
    cmp r4, #0
    beq __pi_digit_loop
    mov r6, #1

__pi_write_digit:
    add r4, r4, #48
    strb r4, [r1], #1
    b __pi_digit_loop

__pi_last_digit:
    add r2, r2, #48
    strb r2, [r1], #1

    mov r7, #4      @ write
    mov r0, #1      @ stdout
    ldr r1, =__print_buf
    mov r2, #12
    svc 0

    pop {r1-r7, pc}


__pi_is_min_int:
    mov r7, #4              @ syscall write
    mov r0, #1              
    ldr r1, =__min_int_str  
    mov r2, #11             @ longueur de la chaîne
    svc 0
    pop {r1-r7, pc}
""");
        
    }

    public void ensurePrintFloat() {
        if (printFloatAdded) return;
        printFloatAdded = true;

        ensurePrintInt();

        addData("__float_10: .float 10.0");

        addRaw("""
__print_float:
    push {r0-r8, lr}
    vpush {s1-s5}

    vmov.f32 s1, s0
    vcmp.f32 s1, #0.0
    vmrs APSR_nzcv, fpscr
    bge __pf_positive

    mov r0, #45
    push {r0}
    mov r7, #4
    mov r0, #1
    pop {r1}
    mov r2, #1
    svc 0

    vneg.f32 s1, s1

__pf_positive:
    vmov.f32 s5, s1
    vcvt.s32.f32 s2, s1
    vmov r0, s2
    bl __print_int
    vmov.f32 s1, s5

    mov r0, #46
    push {r0}
    mov r7, #4
    mov r0, #1
    pop {r1}
    mov r2, #1
    svc 0

    vcvt.s32.f32 s2, s1
    vcvt.f32.s32 s2, s2
    vsub.f32 s1, s1, s2

    mov r4, #6
    ldr r8, =__float_10
    vldr.f32 s4, [r8]

__pf_frac_loop:
    vmul.f32 s1, s1, s4
    vcvt.s32.f32 s3, s1
    vmov r0, s3
    add r0, r0, #48

    push {r0}
    mov r7, #4
    mov r0, #1
    pop {r1}
    mov r2, #1
    svc 0

    vcvt.f32.s32 s3, s3
    vsub.f32 s1, s1, s3

    subs r4, r4, #1
    bne __pf_frac_loop

    vpop {s1-s5}
    pop {r0-r8, pc}
""");

    }

    public void ensurePrintString() {
        if (printStringAdded) return;
        printStringAdded = true;

        addRaw("""
__print_string:
    push {r0-r3, lr}

    mov r1, r0      
    mov r2, #0      

__ps_len:
    ldrb r3, [r1, r2]
    cmp r3, #0
    beq __ps_write
    add r2, r2, #1
    b __ps_len

__ps_write:
    mov r7, #4     
    mov r0, #1      
    svc 0

    pop {r0-r3, pc}
""");
    }

    // runtime pour la division 

    public void ensureDiv() {
        if (divAdded) return;
        divAdded = true;

        addData("division_zero_msg : .ascii \"Erreur : division par zero\\n\"");
        addRaw("""
    __div:
        push {r2-r4, lr}

        cmp r0, #0
        beq division_zero

        mov r2, #0
        mov r3, r1

    __div_loop:
        cmp r3, r0
        blt __div_end
        sub r3, r3, r0
        add r2, r2, #1
        b __div_loop

    __div_end:
        mov r0, r2
        pop {r2-r4, pc}

    division_zero:
        mov r7, #4
        mov r0, #2
        ldr r1, =division_zero_msg
        mov r2, #26
        svc 0

        mov r7, #1
        mov r0, #1
        svc 0
    """);

    }

    public void ensurePrintLn() {
        if (printLnAdded) return;
        printLnAdded = true;

        addData("__newline: .ascii \"\\n\"");

        addRaw("""
    __print_ln:
        push {r0-r2, lr}
            
        mov r7, #4
        mov r0, #1
        ldr r1, =__newline
        mov r2, #1
        svc 0
            
        pop {r0-r2, pc}
    """);
    }

    public void ensureReadInt() {
        if (readIntAdded) return;
        readIntAdded = true;

        addData("__read_buf: .space 12");

        addRaw("""
    __read_int:
        push {r1-r7, lr}

        mov r7, #3
        mov r0, #0
        ldr r1, =__read_buf
        mov r2, #12
        svc 0

        ldr r1, =__read_buf
        mov r0, #0
        mov r2, #1

        ldrb r3, [r1]
        cmp r3, #45
        bne __ri_parse
        mov r2, #-1
        add r1, r1, #1

    __ri_parse:
        ldrb r3, [r1], #1
        cmp r3, #10
        beq __ri_done
        cmp r3, #48
        blt __ri_done
        cmp r3, #57
        bgt __ri_done

        sub r3, r3, #48
        mov r4, #10
        mul r0, r0, r4
        add r0, r0, r3
        b __ri_parse

    __ri_done:
        mul r0, r0, r2
        pop {r1-r7, pc}
    """);
    }

    public void ensureReadFloat() {
        if (readFloatAdded) return;
        readFloatAdded = true;

        ensureReadInt();

        addRaw("""
    __read_float:
        push {r0, lr}

        bl __read_int
        vmov s0, r0
        vcvt.f32.s32 s0, s0

        pop {r0, pc}
    """);
    }

    public void display(PrintStream out) {
        if (!dataSection.isEmpty()) {
            out.println(".data");
            for (String data : dataSection) {
                out.println(data);
            }
            out.println();
        }

        for (String line : textPreamble) {
            out.println(line);
        }


        for (ARMInstruction instr : instructions) {
            out.println("\t" + instr);
        }


        for (String line : rawText) {
            out.println(line);
        }
    }

    public String display() {
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        display(new PrintStream(baos));
        return baos.toString();
    }
}