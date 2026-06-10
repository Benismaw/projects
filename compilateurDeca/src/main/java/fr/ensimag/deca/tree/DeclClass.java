package fr.ensimag.deca.tree;

import fr.ensimag.deca.context.*;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.deca.tools.SymbolTable.Symbol;

import java.io.PrintStream;

import fr.ensimag.deca.context.EnvironmentExp.DoubleDefException;
import fr.ensimag.ima.pseudocode.*;
import fr.ensimag.ima.pseudocode.instructions.*;

/**
 * Declaration of a class (<code>class name extends superClass {members}<code>).
 * 
 * @author gl16
 * @date 01/01/2026
 */
public class DeclClass extends AbstractDeclClass {
    private AbstractIdentifier className;// L'identificateur du nom de la classe 
    private AbstractIdentifier superClass;// L'identificateur du nom de la classe 
    private ListDeclField fields; //les champs 
    private ListDeclMethod methods; //les méthodes 

    public DeclClass(AbstractIdentifier className, AbstractIdentifier superClass,
                     ListDeclField fields, ListDeclMethod methods) {
        this.className = className;
        this.superClass = superClass;
        this.fields = fields;
        this.methods = methods;
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("class ");
        className.decompile(s);
        if (superClass != null) {
            s.print(" extends ");
            superClass.decompile(s);
        }
        s.println(" {");
        s.indent();
        fields.decompile(s);
        methods.decompile(s);
        s.unindent();
        s.println("}");
    }

    @Override
    protected void verifyClass(DecacCompiler compiler) throws ContextualError {
        Symbol classSym = this.className.getName();
        ClassDefinition superDef;

        // Déterminer la super-classe
        if (this.superClass != null) {
            // Recherche dans l'environnement global (Passe 1)
            TypeDefinition def = compiler.environmentType.defOfType(this.superClass.getName());

            if (def == null) {
                // L’identificateur super doit être préalablement déclaré
                throw new ContextualError(
                        "La super-classe " + this.superClass.getName() + " n'est pas encore déclarée",
                        this.superClass.getLocation()
                );
            }
            if (!def.isClass()) {
                throw new ContextualError("La super-classe doit être une classe", this.superClass.getLocation());
            }
            superDef = (ClassDefinition) def;
            this.superClass.setDefinition(superDef);

        } else {
            // Par défaut, la super-classe est Object
            superDef = (ClassDefinition) compiler.environmentType.defOfType(compiler.createSymbol("Object"));
        }

        // Créer le type et la définition
        ClassType classType = new ClassType(classSym, this.getLocation(), superDef);
        ClassDefinition classDef = classType.getDefinition();

        // Ajouter la classe à l'environnement
        compiler.environmentType.declare(classSym, classDef);
        this.className.setDefinition(classDef);
    }


    @Override
    protected void verifyClassMembers(DecacCompiler compiler)
            throws ContextualError {
        ClassDefinition currentClassDef = (ClassDefinition) compiler.environmentType
                .defOfType(this.className.getName());

        ClassDefinition superClassDef = currentClassDef.getSuperClass();

        if (superClassDef != null) {
            currentClassDef.setNumberOfMethods(superClassDef.getNumberOfMethods());
            currentClassDef.getMembers().setParentEnvironment(superClassDef.getMembers());
        }

        // Vérification des champs et des méthodes
        this.fields.verifyListField(compiler, superClassDef, currentClassDef);
        this.methods.verifyListMethod(compiler, superClassDef, currentClassDef);

    }

    @Override
    protected void verifyClassBody(DecacCompiler compiler) throws ContextualError {
        ClassDefinition currentClassDef = (ClassDefinition) compiler.environmentType
                .defOfType(this.className.getName());
        this.fields.verifyListFieldBody(compiler, currentClassDef);
        this.methods.verifyListMethodBody(compiler, currentClassDef);
    }

    public void codeGenVTable(DecacCompiler compiler) {
        ClassDefinition def = className.getClassDefinition();
        ClassDefinition superDef = def.getSuperClass();

        // Calculer la taille totale : 1 (super) + nombre de méthodes total
        int totalMethods = def.getNumberOfMethods();
        int size = 1 + totalMethods;

        // Allouer de la place contiguë dans la Pile Globale (GB)
        // On alloue le premier slot et on mémorise l'adresse de début
        DAddr vtableAddr = compiler.getStackManager().allocGlobalVar();
        def.setVtableAddr(vtableAddr);

        // On alloue les slots restants pour que le StackManager soit à jour
        for (int i = 1; i < size; i++) {
            compiler.getStackManager().allocGlobalVar();
        }

        // Charger l'adresse de la VTable courante dans R1 pour l'adressage indexé
        compiler.addInstruction(new LEA(vtableAddr, Register.R1));

        // Initialiser le lien vers la Super Classe (Index 0)
        if (superDef == null) {
            compiler.addInstruction(new LOAD(new NullOperand(), Register.R0));
        } else {
            // On met l'adresse de la VTable du père (ex: 1(GB)) dans R0
            compiler.addInstruction(new LEA(superDef.getVtableAddr(), Register.R0));
        }
        compiler.addInstruction(new STORE(Register.R0, new RegisterOffset(0, Register.R1)));

        // Copier les méthodes héritées (Si on a un père)
        if (superDef != null) {
            int nbSuperMethods = superDef.getNumberOfMethods();
            for (int i = 1; i <= nbSuperMethods; i++) {
                // LOAD offset(R0), R2 -> Charge l'adresse de la méthode du père
                compiler.addInstruction(new LOAD(new RegisterOffset(i, Register.R0), Register.getR(2)));
                // STORE R2, offset(R1) -> Copie dans la VTable du fils
                compiler.addInstruction(new STORE(Register.getR(2), new RegisterOffset(i, Register.R1)));
            }
        }

        // Ajouter/Ecraser avec les méthodes locales
        for (AbstractDeclMethod m : methods.getList()) {
            int index = m.getMethodDefinition().getIndex();
            Label label = m.getNomMethod().getMethodDefinition().getLabel();

            compiler.addInstruction(new LOAD(new LabelOperand(label), Register.R0));
            compiler.addInstruction(new STORE(Register.R0, new RegisterOffset(index, Register.R1)));
        }
    }

    public void codeGenMethod(DecacCompiler compiler) {
        for (AbstractDeclMethod methode : methods.getList()) {
            methode.codeGenDeclMethod(compiler, this.className.getName().getName());
        }
    }

    public void codeGenInit(DecacCompiler compiler) {
        compiler.addLabel(new Label("init." + className.getName()));
        // Test de débordement de pile
        // d1 dépend du nombre de registres sauvegardés et d'empilements
        compiler.addInstruction(new TSTO(3));
        compiler.addInstruction(new BOV(new Label("pile_pleine")));

        // Récupérer l'adresse de 'this' située à -2(LB)
        compiler.addInstruction(new LOAD(new RegisterOffset(-2, Register.LB), Register.R1));

        // Mise à zéro par défaut des champs propres (prudence sémantique)
        for (AbstractDeclField f : fields.getList()) {
            f.codeGenDefaultInit(compiler); // Utilise R1 comme base
        }
        // Initialiser les champs hérités
        if (superClass != null && !superClass.getName().getName().equals("Object")) {
            compiler.addInstruction(new PUSH(Register.R1));
            compiler.addInstruction(new BSR(new Label("init." + superClass.getName())));
            compiler.addInstruction(new SUBSP(1));
        }

        // Initialiser les champs locaux
        for (AbstractDeclField f : fields.getList()) {
            f.codeGenExplicitInit(compiler);
        }
        compiler.addInstruction(new RTS());
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        //Affiche le nom de la classe(n'est pas le dernier fils)
        this.className.prettyPrint(s, prefix, false);

        // Affiche la super-classe (n'est pas le dernier fils)
        if (this.superClass != null) {
            this.superClass.prettyPrint(s, prefix, false);
        }

        // Affiche la liste des champs (n'est pas le dernier fils)
        this.fields.prettyPrint(s, prefix, false);

        // Affiche la liste des méthodes (C'est le dernier fils)
        this.methods.prettyPrint(s, prefix, true);
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        className.iter(f);
        if (superClass != null) superClass.iter(f);
        fields.iter(f);
        methods.iter(f);
    }
}
