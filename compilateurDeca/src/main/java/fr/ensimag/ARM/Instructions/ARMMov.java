package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMMov extends ARMInstruction {
    
    private ARMRegister dst;
    private ARMOperand src;

    public ARMMov(ARMRegister dst, ARMOperand src) {
        this.dst = dst;
        this.src = src;
    }

    @Override
    public String toString() {
        return "mov " + dst + ", " + src;
    }
}
