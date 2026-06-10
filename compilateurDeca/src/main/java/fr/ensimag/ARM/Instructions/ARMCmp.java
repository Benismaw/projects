package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMInstruction;
import fr.ensimag.ARM.ARMOperand;
import fr.ensimag.ARM.ARMRegister;

public class ARMCmp extends ARMInstruction {

    private ARMRegister r1;
    private ARMOperand op2;

    public ARMCmp(ARMRegister r1, ARMRegister r2) {
        this.r1 = r1;
        this.op2 = r2;
    }

    public ARMCmp(ARMRegister r1, ARMImmediate r2) {
        this.r1 = r1;
        this.op2 = r2;
    }

    @Override
    public String toString() {
        return "cmp " + r1 + ", " + op2;
    }
}