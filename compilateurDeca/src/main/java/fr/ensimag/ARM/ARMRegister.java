package fr.ensimag.ARM;

public class ARMRegister extends ARMOperand {
    
    private String name;

    public ARMRegister(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
    
    //stack pointer and link register
    public static final ARMRegister SP = new ARMRegister("sp");
    public static final ARMRegister LR = new ARMRegister("lr");


    //initialize general purpose registers R0 to R15
    private static final ARMGPRegister[] R = initRegisters();

    public static final ARMGPRegister R0 = R[0];
    public static final ARMGPRegister R1 = R[1];
    public static final ARMGPRegister R2 = R[2];
    public static final ARMGPRegister R3 = R[3];
    public static final ARMGPRegister R4 = R[4];
    public static final ARMGPRegister R5 = R[5];
    public static final ARMGPRegister R6 = R[6];
    public static final ARMGPRegister R7 = R[7];
    public static final ARMGPRegister R8 = R[8];
    public static final ARMGPRegister R9 = R[9];
    public static final ARMGPRegister R10 = R[10];
    public static final ARMGPRegister R11 = R[11]; //frame pointer
    public static final ARMGPRegister R12 = R[12];
    public static final ARMGPRegister R13 = R[13];
    public static final ARMGPRegister R14 = R[14];
    public static final ARMGPRegister R15 = R[15];
    

    // Initialize the registers R0 to R15
    static private ARMGPRegister[] initRegisters() {
        ARMGPRegister [] res = new ARMGPRegister[16];
        for (int i = 0; i < 16; i++) {
            res[i] = new ARMGPRegister("r" + i, i);
        }
        return res;
    }

    public static ARMGPRegister getR(int i) {
        if (i < 0 || i >= 16) {
            throw new IllegalArgumentException("Invalid ARM register index: " + i);
        }
        return R[i];
    }
}

