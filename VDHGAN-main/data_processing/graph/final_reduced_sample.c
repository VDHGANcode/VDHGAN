digraph {
	5 -> 7 [label=IS_AST_PARENT color=black]
	5 -> 10 [label=IS_AST_PARENT color=black]
	5 -> 9 [label=IS_AST_PARENT color=black]
	9 -> 10 [label=IS_AST_PARENT color=black]
	9 -> 11 [label=IS_AST_PARENT color=black]
	11 -> 12 [label=IS_AST_PARENT color=black]
	11 -> 123 [label=IS_AST_PARENT color=black]
	17 -> 19 [label=IS_AST_PARENT color=black]
	17 -> 20 [label=IS_AST_PARENT color=black]
	17 -> 21 [label=IS_AST_PARENT color=black]
	21 -> 132 [label=IS_AST_PARENT color=black]
	21 -> 132 [label=IS_AST_PARENT color=black]
	24 -> 26 [label=IS_AST_PARENT color=black]
	24 -> 27 [label=IS_AST_PARENT color=black]
	24 -> 28 [label=IS_AST_PARENT color=black]
	28 -> 27 [label=IS_AST_PARENT color=black]
	28 -> 30 [label=IS_AST_PARENT color=black]
	30 -> 31 [label=IS_AST_PARENT color=black]
	30 -> 10 [label=IS_AST_PARENT color=black]
	30 -> 20 [label=IS_AST_PARENT color=black]
	38 -> 40 [label=IS_AST_PARENT color=black]
	38 -> 41 [label=IS_AST_PARENT color=black]
	38 -> 42 [label=IS_AST_PARENT color=black]
	42 -> 41 [label=IS_AST_PARENT color=black]
	42 -> 44 [label=IS_AST_PARENT color=black]
	45 -> 47 [label=IS_AST_PARENT color=black]
	45 -> 115 [label=IS_AST_PARENT color=black]
	45 -> 49 [label=IS_AST_PARENT color=black]
	49 -> 41 [label=IS_AST_PARENT color=black]
	49 -> 41 [label=IS_AST_PARENT color=black]
	52 -> 54 [label=IS_AST_PARENT color=black]
	52 -> 55 [label=IS_AST_PARENT color=black]
	52 -> 56 [label=IS_AST_PARENT color=black]
	56 -> 55 [label=IS_AST_PARENT color=black]
	56 -> 58 [label=IS_AST_PARENT color=black]
	59 -> 55 [label=IS_AST_PARENT color=black]
	59 -> 62 [label=IS_AST_PARENT color=black]
	62 -> 63 [label=IS_AST_PARENT color=black]
	62 -> 41 [label=IS_AST_PARENT color=black]
	62 -> 68 [label=IS_AST_PARENT color=black]
	62 -> 72 [label=IS_AST_PARENT color=black]
	62 -> 74 [label=IS_AST_PARENT color=black]
	62 -> 78 [label=IS_AST_PARENT color=black]
	62 -> 82 [label=IS_AST_PARENT color=black]
	62 -> 86 [label=IS_AST_PARENT color=black]
	68 -> 70 [label=IS_AST_PARENT color=black]
	68 -> 71 [label=IS_AST_PARENT color=black]
	74 -> 27 [label=IS_AST_PARENT color=black]
	74 -> 77 [label=IS_AST_PARENT color=black]
	78 -> 27 [label=IS_AST_PARENT color=black]
	78 -> 81 [label=IS_AST_PARENT color=black]
	82 -> 27 [label=IS_AST_PARENT color=black]
	82 -> 85 [label=IS_AST_PARENT color=black]
	86 -> 27 [label=IS_AST_PARENT color=black]
	86 -> 89 [label=IS_AST_PARENT color=black]
	90 -> 92 [label=IS_AST_PARENT color=black]
	90 -> 95 [label=IS_AST_PARENT color=black]
	95 -> 55 [label=IS_AST_PARENT color=black]
	95 -> 98 [label=IS_AST_PARENT color=black]
	98 -> 99 [label=IS_AST_PARENT color=black]
	98 -> 102 [label=IS_AST_PARENT color=black]
	99 -> 100 [label=IS_AST_PARENT color=black]
	99 -> 101 [label=IS_AST_PARENT color=black]
	103 -> 105 [label=IS_AST_PARENT color=black]
	103 -> 108 [label=IS_AST_PARENT color=black]
	103 -> 110 [label=IS_AST_PARENT color=black]
	103 -> 112 [label=IS_AST_PARENT color=black]
	103 -> 116 [label=IS_AST_PARENT color=black]
	112 -> 114 [label=IS_AST_PARENT color=black]
	112 -> 115 [label=IS_AST_PARENT color=black]
	121 -> 122 [label=IS_AST_PARENT color=black]
	121 -> 123 [label=IS_AST_PARENT color=black]
	124 -> 125 [label=IS_AST_PARENT color=black]
	124 -> 108 [label=IS_AST_PARENT color=black]
	127 -> 128 [label=IS_AST_PARENT color=black]
	127 -> 110 [label=IS_AST_PARENT color=black]
	130 -> 131 [label=IS_AST_PARENT color=black]
	130 -> 132 [label=IS_AST_PARENT color=black]
	133 -> 134 [label=IS_AST_PARENT color=black]
	133 -> 116 [label=IS_AST_PARENT color=black]
	136 -> 121 [label=FLOWS_TO color=blue]
	121 -> 124 [label=FLOWS_TO color=blue]
	124 -> 127 [label=FLOWS_TO color=blue]
	127 -> 130 [label=FLOWS_TO color=blue]
	130 -> 133 [label=FLOWS_TO color=blue]
	133 -> 5 [label=FLOWS_TO color=blue]
	5 -> 17 [label=FLOWS_TO color=blue]
	17 -> 24 [label=FLOWS_TO color=blue]
	24 -> 38 [label=FLOWS_TO color=blue]
	38 -> 45 [label=FLOWS_TO color=blue]
	45 -> 52 [label=FLOWS_TO color=blue]
	52 -> 59 [label=FLOWS_TO color=blue]
	59 -> 90 [label=FLOWS_TO color=blue]
	90 -> 103 [label=FLOWS_TO color=blue]
	103 -> 137 [label=FLOWS_TO color=blue]
	59 -> 90 [label=REACHES color=red]
	130 -> 17 [label=REACHES color=red]
	5 -> 24 [label=REACHES color=red]
	133 -> 103 [label=REACHES color=red]
	121 -> 5 [label=REACHES color=red]
	124 -> 103 [label=REACHES color=red]
	17 -> 24 [label=REACHES color=red]
	127 -> 103 [label=REACHES color=red]
	38 -> 45 [label=REACHES color=red]
	45 -> 59 [label=REACHES color=red]
	24 -> 59 [label=REACHES color=red]
	122 -> 123 [label=NSC color=green]
	123 -> 125 [label=NSC color=green]
	125 -> 108 [label=NSC color=green]
	108 -> 128 [label=NSC color=green]
	128 -> 110 [label=NSC color=green]
	110 -> 131 [label=NSC color=green]
	131 -> 132 [label=NSC color=green]
	132 -> 134 [label=NSC color=green]
	134 -> 116 [label=NSC color=green]
	116 -> 7 [label=NSC color=green]
	7 -> 10 [label=NSC color=green]
	10 -> 12 [label=NSC color=green]
	12 -> 19 [label=NSC color=green]
	19 -> 20 [label=NSC color=green]
	20 -> 26 [label=NSC color=green]
	26 -> 27 [label=NSC color=green]
	27 -> 31 [label=NSC color=green]
	31 -> 40 [label=NSC color=green]
	40 -> 41 [label=NSC color=green]
	41 -> 44 [label=NSC color=green]
	44 -> 47 [label=NSC color=green]
	47 -> 115 [label=NSC color=green]
	115 -> 54 [label=NSC color=green]
	54 -> 55 [label=NSC color=green]
	55 -> 58 [label=NSC color=green]
	58 -> 63 [label=NSC color=green]
	63 -> 70 [label=NSC color=green]
	70 -> 71 [label=NSC color=green]
	71 -> 72 [label=NSC color=green]
	72 -> 77 [label=NSC color=green]
	77 -> 81 [label=NSC color=green]
	81 -> 85 [label=NSC color=green]
	85 -> 89 [label=NSC color=green]
	89 -> 92 [label=NSC color=green]
	92 -> 100 [label=NSC color=green]
	100 -> 101 [label=NSC color=green]
	101 -> 102 [label=NSC color=green]
	102 -> 105 [label=NSC color=green]
	105 -> 114 [label=NSC color=green]
	68 [label="68
sizeof ( buffer )
Argument"]
	125 [label="125
Visitor *
ParameterType"]
	103 [label="103
\"visit_type_str ( v , name , & p , errp )\"
ExpressionStatement"]
	5 [label="5
DeviceState * dev = DEVICE ( obj ) ;
IdentifierDeclStatement"]
	17 [label="17
Property * prop = opaque ;
IdentifierDeclStatement"]
	31 [label="31
qdev_get_prop_ptr
Callee"]
	72 [label="72
\"\"\"%04x:%02x:%02x.%d\"\"\"
Argument"]
	116 [label="116
errp
Argument"]
	52 [label="52
int rc = 0 ;
IdentifierDeclStatement"]
	98 [label="98
sizeof ( buffer ) - 1
AdditiveExpression"]
	105 [label="105
visit_type_str
Callee"]
	20 [label="20
prop
Identifier"]
	114 [label="114
&
UnaryOperator"]
	124 [label="124
Visitor * v
Parameter"]
	30 [label="30
\"qdev_get_prop_ptr ( dev , prop )\"
CallExpression"]
	85 [label="85
slot
Identifier"]
	12 [label="12
DEVICE
Callee"]
	127 [label="127
const char * name
Parameter"]
	54 [label="54
int
IdentifierDeclType"]
	58 [label="58
0
PrimaryExpression"]
	100 [label="100
sizeof
Sizeof"]
	122 [label="122
Object *
ParameterType"]
	115 [label="115
p
Identifier"]
	62 [label="62
\"snprintf ( buffer , sizeof ( buffer ) , \"\"%04x:%02x:%02x.%d\"\" , addr -> domain , addr -> bus , addr -> slot , addr -> function )\"
CallExpression"]
	101 [label="101
buffer
SizeofOperand"]
	9 [label="9
* dev = DEVICE ( obj )
AssignmentExpression"]
	26 [label="26
PCIHostDeviceAddress *
IdentifierDeclType"]
	77 [label="77
domain
Identifier"]
	55 [label="55
rc
Identifier"]
	110 [label="110
name
Argument"]
	128 [label="128
const char *
ParameterType"]
	11 [label="11
DEVICE ( obj )
CallExpression"]
	70 [label="70
sizeof
Sizeof"]
	134 [label="134
Error * *
ParameterType"]
	47 [label="47
char *
IdentifierDeclType"]
	19 [label="19
Property *
IdentifierDeclType"]
	21 [label="21
* prop = opaque
AssignmentExpression"]
	82 [label="82
addr -> slot
Argument"]
	81 [label="81
bus
Identifier"]
	78 [label="78
addr -> bus
Argument"]
	102 [label="102
1
PrimaryExpression"]
	130 [label="130
void * opaque
Parameter"]
	136 [label="136
ENTRY
CFGEntryNode"]
	137 [label="137
EXIT
CFGExitNode"]
	10 [label="10
dev
Identifier"]
	86 [label="86
addr -> function
Argument"]
	89 [label="89
function
Identifier"]
	7 [label="7
DeviceState *
IdentifierDeclType"]
	95 [label="95
rc == sizeof ( buffer ) - 1
Argument"]
	42 [label="42
\"buffer [ ] = \"\"xxxx:xx:xx.x\"\"\"
AssignmentExpression"]
	44 [label="44
\"\"\"xxxx:xx:xx.x\"\"\"
PrimaryExpression"]
	99 [label="99
sizeof ( buffer )
SizeofExpression"]
	74 [label="74
addr -> domain
Argument"]
	40 [label="40
char [ ]
IdentifierDeclType"]
	112 [label="112
& p
Argument"]
	41 [label="41
buffer
Identifier"]
	121 [label="121
Object * obj
Parameter"]
	92 [label="92
assert
Callee"]
	63 [label="63
snprintf
Callee"]
	27 [label="27
addr
Identifier"]
	71 [label="71
buffer
SizeofOperand"]
	49 [label="49
* p = buffer
AssignmentExpression"]
	24 [label="24
\"PCIHostDeviceAddress * addr = qdev_get_prop_ptr ( dev , prop ) ;\"
IdentifierDeclStatement"]
	56 [label="56
rc = 0
AssignmentExpression"]
	45 [label="45
char * p = buffer ;
IdentifierDeclStatement"]
	108 [label="108
v
Argument"]
	133 [label="133
Error * * errp
Parameter"]
	123 [label="123
obj
Identifier"]
	28 [label="28
\"* addr = qdev_get_prop_ptr ( dev , prop )\"
AssignmentExpression"]
	90 [label="90
assert ( rc == sizeof ( buffer ) - 1 )
ExpressionStatement"]
	38 [label="38
\"char buffer [ ] = \"\"xxxx:xx:xx.x\"\" ;\"
IdentifierDeclStatement"]
	131 [label="131
void *
ParameterType"]
	132 [label="132
opaque
Identifier"]
	59 [label="59
\"rc = snprintf ( buffer , sizeof ( buffer ) , \"\"%04x:%02x:%02x.%d\"\" , addr -> domain , addr -> bus , addr -> slot , addr -> function )\"
ExpressionStatement"]
}
