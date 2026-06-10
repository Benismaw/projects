package fr.ensimag.ARM.Instructions;

import fr.ensimag.ARM.*;

public class ARMAdd extends ARMInstruction {
    
    private ARMRegister dst, src1, src2;

    public ARMAdd(ARMRegister dst, ARMRegister src1, ARMRegister src2) {
        this.dst = dst;
        this.src1 = src1;
        this.src2 = src2;
    }

    @Override
    public String toString() {
        return "add " + dst + ", " + src1 + ", " + src2;
    }
}
