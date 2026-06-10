package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMLabelOperand;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMLdr;
import fr.ensimag.ARM.Instructions.ARMStr;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.STORE;

/**
 * Assignment, i.e. lvalue = expr.
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Assign extends AbstractBinaryExpr {

    @Override
    public AbstractLValue getLeftOperand() {
        // The cast succeeds by construction, as the leftOperand has been set
        // as an AbstractLValue by the constructor.
        return (AbstractLValue)super.getLeftOperand();
    }

    public Assign(AbstractLValue leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        // On vérifie chaque operande
        Type typeLeft = getLeftOperand().verifyExpr(compiler, localEnv, currentClass);

        // Regle 3.28
        AbstractExpr rightExpr = getRightOperand().verifyRValue(compiler, localEnv, currentClass, typeLeft);

        setRightOperand(rightExpr);

        setType(typeLeft);
        return typeLeft;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        // On calcul l'operande de droite
        getRightOperand().codeGenExpr(compiler, register);

        // On récupère l'adresse de l'operande gauche
        AbstractLValue leftOperand = getLeftOperand();

        // CAS 1 : Identifiant
        if (leftOperand instanceof Identifier) {
            Identifier identifier = (Identifier) leftOperand;

            // Si c'est un champ de la classe 
            if (identifier.getDefinition().isField()) {
                // On a besoin de charger l'adresse de 'this' (-2(LB))
                GPRegister addrReg = compiler.getRegManager().allocate(compiler);

                boolean allocatedTwice = false;

                // Si le RegManager nous donne le registre qui contient déjà le résultat
                if (addrReg.getNumber() == register.getNumber()) {
                    addrReg = compiler.getRegManager().allocate(compiler); // On prend le suivant
                    allocatedTwice = true;
                }
                compiler.addInstruction(new LOAD(new RegisterOffset(-2, Register.LB), addrReg));

                // On stocke dans le champ (offset par rapport à this)
                int index = identifier.getFieldDefinition().getIndex();
                compiler.addInstruction(new STORE(register, new RegisterOffset(index, addrReg)));

                compiler.getRegManager().free(compiler);
                if (allocatedTwice) {
                    compiler.getRegManager().free(compiler);
                }
            }
            // Si c'est une variable locale ou un paramètre
            else {
                DAddr adresseVar = identifier.getExpDefinition().getOperand();
                compiler.addInstruction(new STORE(register, adresseVar));
            }
        }
        // CAS 2 : Sélection
        else if (leftOperand instanceof Selection) {
            Selection sel = (Selection) leftOperand;

            // On a besoin d'un nouveau registre pour l'adresse de l'objet
            GPRegister addrReg = compiler.getRegManager().allocate(compiler);

            boolean allocatedTwice = false;

            if (addrReg.getNumber() == register.getNumber()) {
                addrReg = compiler.getRegManager().allocate(compiler);
                allocatedTwice = true;
            }

            // Calcul de l'adresse de l'objet
            sel.getExpr().codeGenExpr(compiler, addrReg);

            // Vérification nullité
            compiler.addInstruction(new CMP(new NullOperand(), addrReg));
            compiler.addInstruction(new BEQ(new Label("dereferencement_null")));

            // Récupération de l'index du champ
            int index = sel.getFieldDefinition().getIndex();

            // Stockage : Mem[addrReg + index] = register
            compiler.addInstruction(new STORE(register, new RegisterOffset(index, addrReg)));

            // Libération du registre d'adresse
            compiler.getRegManager().free(compiler);
            if (allocatedTwice) {
                compiler.getRegManager().free(compiler);
            }
        }
        else {
            // Impossible
        }
    }

    @Override
    public void decompile(IndentPrintStream s) {
        getLeftOperand().decompile(s);
        s.print(" " + getOperatorName() + " ");
        getRightOperand().decompile(s);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        throw new UnsupportedOperationException("Ne jamais appeler writeOperation sur un Assign");
    }

    @Override
    protected String getOperatorName() {
        return "=";
    }

    @Override
protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {

    var prog = compiler.getARMProgram();
    String name = ((Identifier) getLeftOperand()).getName().getName();
    AbstractExpr rhs = getRightOperand();

    
    if (getLeftOperand().getType().isFloat()) {

        
        rhs.codeGenExprARM(compiler, ARMRegister.R0);

        
        if (rhs.getType().isInt()) {
            prog.addRaw("""
                vmov s0, r0
                vcvt.f32.s32 s0, s0
            """);
        } else {
            prog.addRaw("""
                vmov s0, r0
            """);
        }

        
        prog.addInstruction(
            new ARMLdr(
                ARMRegister.R1,
                new ARMLabelOperand(new ARMLabel("=" + name))
            )
        );
        prog.addRaw("    vstr.f32 s0, [r1]");
        return;
    }

    
    rhs.codeGenExprARM(compiler, ARMRegister.R0);
    prog.addInstruction(
        new ARMLdr(
            ARMRegister.R1,
            new ARMLabelOperand(new ARMLabel("=" + name))
        )
    );
    prog.addInstruction(new ARMStr(ARMRegister.R0, ARMRegister.R1));
}

}
